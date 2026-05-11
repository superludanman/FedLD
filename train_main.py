from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvgWeighted
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import DenseNet121
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader
from semantic_anchor import init_semantic_anchors, get_classifier_layer, reduce_prototypes


def split(ds, num_users):
    num_items = int(len(ds) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(ds))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def eval_top1(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, _, image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            _, logits = model(image)
            pred = torch.argmax(logits, dim=1)
            target = torch.argmax(label, dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    model.train()
    return correct / max(total, 1)


snapshot_path = 'model/'
supervised_user_id = [0, 1]
unsupervised_user_id = [2, 3, 4, 5, 6, 7, 8, 9]
flag_create = False
print('done')

if __name__ == '__main__':
    args = args_parser()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    requested_gpu_ids = [g.strip() for g in args.gpu.split(',') if g.strip() != '']
    visible_gpu_count = torch.cuda.device_count()
    use_cuda = torch.cuda.is_available() and visible_gpu_count > 0
    use_dp = use_cuda and len(requested_gpu_ids) > 1 and visible_gpu_count >= len(requested_gpu_ids)
    device = torch.device("cuda" if use_cuda else "cpu")

    logging.basicConfig(filename="log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("CUDA available: {}, visible GPUs: {}, DataParallel: {}".format(use_cuda, visible_gpu_count, use_dp))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    train_transform = dataset.TransformTwice(transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = dataset.CIFARDataset(root_dir=args.root_path, dataset_name=args.dataset, train=True, transform=train_transform, download=False)
    test_dataset = dataset.CIFARDataset(root_dir=args.root_path, dataset_name=args.dataset, train=False, transform=test_transform, download=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_cuda)

    dict_users = split(train_dataset, args.num_users)
    net_glob = DenseNet121(out_size=args.num_classes, mode=args.label_uncertainty, drop_rate=args.drop_rate)
    net_glob = net_glob.to(device)

    if use_dp:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=list(range(len(requested_gpu_ids))))
    net_glob.train()

    anchor_dim = get_classifier_layer(net_glob).weight.size(1)
    anchor_bank = init_semantic_anchors(args.num_classes, anchor_dim, device=device)
    w_glob = net_glob.state_dict()
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, train_dataset, dict_users[i]))
        net_locals.append(copy.deepcopy(net_glob).to(device))
        optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        trainer_locals.append(UnsupervisedLocalUpdate(args, train_dataset, dict_users[i]))

    for com_round in range(args.rounds):
        print("begin")
        loss_locals = []
        proto_sums = torch.zeros_like(anchor_bank)
        proto_counts = torch.zeros(args.num_classes, device=anchor_bank.device)
        current_weights = []
        current_coeffs = []
        unsup_confidence_log = []

        for idx in supervised_user_id:
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            w, loss, op, local_proto_sum, local_proto_count = local.train(args, net_locals[idx], optimizer, anchor_bank.detach())
            current_weights.append(copy.deepcopy(w))
            current_coeffs.append(1.0)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
            proto_sums += local_proto_sum.to(anchor_bank.device)
            proto_counts += local_proto_count.to(anchor_bank.device)

        if not flag_create:
            print('begin unsup')
            for i in unsupervised_user_id:
                net_locals.append(copy.deepcopy(net_glob).to(device))
                optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
                optim_locals.append(copy.deepcopy(optimizer.state_dict()))
            flag_create = True

        for idx in unsupervised_user_id:
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            w, loss, op, unsup_confidence, unsup_proto_sum, unsup_proto_count = local.train(
                args, net_locals[idx], optimizer, com_round * args.local_ep, anchor_bank.detach()
            )
            conf_clamped = min(max(float(unsup_confidence), 0.0), 1.0)
            unsup_coeff = args.unsup_min_weight + (args.unsup_max_weight - args.unsup_min_weight) * conf_clamped
            current_weights.append(copy.deepcopy(w))
            current_coeffs.append(float(unsup_coeff))
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
            unsup_confidence_log.append(conf_clamped)
            if conf_clamped >= args.unsup_conf_for_anchor:
                proto_sums += unsup_proto_sum.to(anchor_bank.device)
                proto_counts += unsup_proto_count.to(anchor_bank.device)

        with torch.no_grad():
            proto_mean, valid = reduce_prototypes(proto_sums, proto_counts)
            if valid.any():
                anchor_bank[valid] = args.anchor_momentum * anchor_bank[valid] + (1 - args.anchor_momentum) * proto_mean[valid]
                anchor_bank = F.normalize(anchor_bank, dim=1)

        with torch.no_grad():
            w_glob = FedAvgWeighted(current_weights, current_coeffs)

        net_glob.load_state_dict(w_glob)
        for i in supervised_user_id:
            net_locals[i].load_state_dict(w_glob)
        for i in unsupervised_user_id:
            net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print(loss_avg, com_round)
        logging.info('Loss Avg {} Round {} LR {} '.format(loss_avg, com_round, args.base_lr))
        if len(unsup_confidence_log) > 0:
            logging.info('Unsup confidence mean {:.4f} max {:.4f}'.format(
                float(np.mean(unsup_confidence_log)),
                float(np.max(unsup_confidence_log))
            ))
        if com_round % 10 == 0:
            os.makedirs(snapshot_path, exist_ok=True)
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(com_round) + '.pth')
            state_dict = net_glob.module.state_dict() if hasattr(net_glob, 'module') else net_glob.state_dict()
            torch.save({'state_dict': state_dict}, save_mode_path)
            top1 = eval_top1(net_glob, test_dataloader, device)
            logging.info("\nTEST Round {}: Top-1 Acc {:.6f}".format(com_round, top1))
