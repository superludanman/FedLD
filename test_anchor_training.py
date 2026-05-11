import copy
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_anchor import (
    anchor_calibration_loss,
    compute_batch_prototypes,
    init_semantic_anchors,
    reduce_prototypes,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ToyDenseBackbone(nn.Module):
    def __init__(self, in_dim, feat_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return feat, logits


class ToyModel(nn.Module):
    """
    Keep the naming convention compatible with semantic_anchor.get_classifier_layer:
    model.densenet121.classifier[0]
    """

    def __init__(self, in_dim, feat_dim, num_classes):
        super().__init__()
        self.densenet121 = ToyDenseBackbone(in_dim, feat_dim, num_classes)

    def forward(self, x):
        return self.densenet121(x)


def make_labeled_batch(batch_size, in_dim, num_classes, device):
    labels = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)
    centers = torch.linspace(-1.5, 1.5, steps=num_classes, device=device).unsqueeze(1).repeat(1, in_dim)
    noise = 0.4 * torch.randn(batch_size, in_dim, device=device)
    x = centers[labels] + noise
    y = F.one_hot(labels, num_classes=num_classes).float()
    return x, y


def make_unlabeled_batch(batch_size, in_dim, num_classes, device):
    pseudo_true = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)
    centers = torch.linspace(-1.5, 1.5, steps=num_classes, device=device).unsqueeze(1).repeat(1, in_dim)
    x = centers[pseudo_true] + 0.6 * torch.randn(batch_size, in_dim, device=device)
    return x


def update_ema(model, ema_model, alpha=0.99):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(alpha).add_(p.data, alpha=1 - alpha)


def run_test():
    set_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SimpleNamespace(
        num_classes=5,
        in_dim=32,
        feat_dim=24,
        sup_batch=80,
        unsup_batch=80,
        lr=1e-3,
        anchor_momentum=0.9,
        anchor_proto_weight=1.0,
        anchor_calib_weight=0.1,
        anchor_align_weight=1.0,
        anchor_pseudo_weight=1.0,
        anchor_logit_weight=0.5,
        pseudo_threshold=0.2,
    )

    model = ToyModel(cfg.in_dim, cfg.feat_dim, cfg.num_classes).to(device)
    ema_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    anchor_bank = init_semantic_anchors(cfg.num_classes, cfg.feat_dim, device=device)

    # -------------------------
    # 1) Supervised client step
    # -------------------------
    model.train()
    x_sup, y_sup = make_labeled_batch(cfg.sup_batch, cfg.in_dim, cfg.num_classes, device)
    feat_sup, logits_sup = model(x_sup)
    cls_loss = F.cross_entropy(logits_sup, torch.argmax(y_sup, dim=1))

    proto_sum, proto_count = compute_batch_prototypes(feat_sup, y_sup, cfg.num_classes)
    proto_mean, valid = reduce_prototypes(proto_sum, proto_count)

    if valid.any():
        proto_reg = F.mse_loss(
            F.normalize(proto_mean[valid], dim=1),
            F.normalize(anchor_bank[valid], dim=1),
        )
    else:
        proto_reg = torch.tensor(0.0, device=device)

    calib_loss = anchor_calibration_loss(model, anchor_bank.detach())
    sup_loss = cls_loss + cfg.anchor_proto_weight * proto_reg + cfg.anchor_calib_weight * calib_loss
    optimizer.zero_grad()
    sup_loss.backward()
    optimizer.step()

    # Server anchor EMA update
    old_anchor = anchor_bank.clone()
    with torch.no_grad():
        if valid.any():
            anchor_bank[valid] = cfg.anchor_momentum * anchor_bank[valid] + (1 - cfg.anchor_momentum) * proto_mean[valid]
            anchor_bank = F.normalize(anchor_bank, dim=1)

    # Check: anchor moved closer to prototype for valid classes
    improved = True
    if valid.any():
        old_sim = F.cosine_similarity(old_anchor[valid], proto_mean[valid], dim=1).mean().item()
        new_sim = F.cosine_similarity(anchor_bank[valid], proto_mean[valid], dim=1).mean().item()
        improved = new_sim >= old_sim - 1e-6
    else:
        old_sim = float("nan")
        new_sim = float("nan")

    # ---------------------------
    # 2) Unsupervised client step
    # ---------------------------
    model.train()
    x_unsup = make_unlabeled_batch(cfg.unsup_batch, cfg.in_dim, cfg.num_classes, device)
    with torch.no_grad():
        ema_feat, ema_logits = ema_model(x_unsup)

    feat_unsup, logits_unsup = model(x_unsup)
    consistency = torch.mean((F.softmax(logits_unsup, dim=1) - F.softmax(ema_logits, dim=1)) ** 2)

    anchor_scores = torch.matmul(
        F.normalize(feat_unsup.detach(), dim=1),
        F.normalize(anchor_bank.detach(), dim=1).t(),
    )
    combined_logits = logits_unsup + cfg.anchor_logit_weight * anchor_scores
    combined_probs = F.softmax(combined_logits, dim=1)
    confidence, pseudo_idx = torch.max(combined_probs, dim=1)
    mask = confidence >= cfg.pseudo_threshold
    pseudo_count = int(mask.sum().item())

    if mask.any():
        pseudo_loss = F.cross_entropy(logits_unsup[mask], pseudo_idx[mask].long())
        align_loss = F.mse_loss(
            F.normalize(feat_unsup[mask], dim=1),
            F.normalize(anchor_bank[pseudo_idx[mask].long()].detach(), dim=1),
        )
    else:
        pseudo_loss = torch.tensor(0.0, device=device)
        align_loss = torch.tensor(0.0, device=device)

    calib_unsup = anchor_calibration_loss(model, anchor_bank.detach())
    unsup_loss = consistency + cfg.anchor_pseudo_weight * pseudo_loss + cfg.anchor_align_weight * align_loss + cfg.anchor_calib_weight * calib_unsup

    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad()
    unsup_loss.backward()
    optimizer.step()
    update_ema(model, ema_model, alpha=0.99)
    after = [p.detach().clone() for p in model.parameters()]
    changed = any([(b - a).abs().sum().item() > 0 for b, a in zip(before, after)])

    # -------------------------
    # Final report
    # -------------------------
    print("=" * 60)
    print("Semantic Anchor Pipeline Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"[Supervised] loss={sup_loss.item():.6f}, valid_classes={int(valid.sum().item())}")
    print(f"[Anchor EMA] cos_sim old={old_sim:.6f}, new={new_sim:.6f}, improved={improved}")
    print(f"[Unsupervised] loss={unsup_loss.item():.6f}, pseudo_count={pseudo_count}, param_changed={changed}")

    ok_anchor = bool(improved)
    ok_unsup = bool(pseudo_count > 0 and changed and torch.isfinite(unsup_loss))

    print("-" * 60)
    print(f"CHECK 1 (anchor update): {'PASS' if ok_anchor else 'FAIL'}")
    print(f"CHECK 2 (unsup anchor pseudo training): {'PASS' if ok_unsup else 'FAIL'}")
    print("=" * 60)

    if ok_anchor and ok_unsup:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        raise RuntimeError("Semantic anchor test failed.")


if __name__ == "__main__":
    run_test()
