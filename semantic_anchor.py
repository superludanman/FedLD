import torch
import torch.nn.functional as F


def get_classifier_layer(model):
    base_model = model.module if hasattr(model, "module") else model
    if getattr(base_model.densenet121, "classifier", None) is not None:
        classifier = base_model.densenet121.classifier
        return classifier[0] if isinstance(classifier, torch.nn.Sequential) else classifier
    if hasattr(base_model.densenet121, "Linear_0"):
        return base_model.densenet121.Linear_0
    raise AttributeError("No classifier layer found for semantic anchor calibration.")


def init_semantic_anchors(num_classes, feat_dim, device):
    anchors = torch.randn(num_classes, feat_dim, device=device)
    return F.normalize(anchors, dim=1)


def compute_batch_prototypes(features, onehot_labels, num_classes):
    proto_sum = features.new_zeros((num_classes, features.size(1)))
    proto_count = features.new_zeros((num_classes,))
    label_indices = torch.argmax(onehot_labels, dim=1)
    for class_id in range(num_classes):
        class_mask = label_indices == class_id
        if class_mask.any():
            class_feats = features[class_mask]
            proto_sum[class_id] = class_feats.sum(dim=0)
            proto_count[class_id] = class_mask.float().sum()
    return proto_sum, proto_count


def reduce_prototypes(proto_sum, proto_count):
    prototypes = proto_sum.clone()
    valid = proto_count > 0
    if valid.any():
        prototypes[valid] = proto_sum[valid] / proto_count[valid].unsqueeze(1)
    return prototypes, valid


def anchor_calibration_loss(model, anchors):
    classifier = get_classifier_layer(model)
    logits = classifier(anchors)
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return F.cross_entropy(logits, labels)
