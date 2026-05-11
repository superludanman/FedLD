import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import args_parser
from networks.models import DenseNet121
from dataloaders import dataset


def main():
    args = args_parser()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100

    checkpoint_path = os.path.join('model', 'epoch_0.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DenseNet121(out_size=args.num_classes, mode=args.label_uncertainty, drop_rate=args.drop_rate).to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    test_dataset = dataset.CIFARDataset(
        root_dir=args.root_path,
        dataset_name=args.dataset,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    correct, total = 0, 0
    with torch.no_grad():
        for _, _, image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            _, logits = net(image)
            pred = torch.argmax(logits, dim=1)
            gt = torch.argmax(label, dim=1)
            correct += int((pred == gt).sum().item())
            total += int(gt.numel())

    acc = correct / max(total, 1)
    print("Top-1 Accuracy:", acc)


if __name__ == "__main__":
    main()
