# encoding: utf-8
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class CIFARDataset(Dataset):
    """
    Output format is aligned with original project:
    returns: items, index, image, onehot_label
    where image can be either tensor or (tensor, tensor) when TransformTwice is used.
    """

    def __init__(self, root_dir, dataset_name="cifar10", train=True, transform=None, download=False):
        super(CIFARDataset, self).__init__()
        dataset_name = dataset_name.lower()
        if dataset_name == "cifar10":
            dataset_ctor = datasets.CIFAR10
            self.num_classes = 10
            subdirs = ["CIFAR10", "cifar10", "cifar-10"]
        elif dataset_name == "cifar100":
            dataset_ctor = datasets.CIFAR100
            self.num_classes = 100
            subdirs = ["CIFAR100", "cifar100", "cifar-100"]
        else:
            raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

        candidate_roots = [root_dir] + [os.path.join(root_dir, s) for s in subdirs]
        base = None
        last_err = None
        for cand_root in candidate_roots:
            try:
                base = dataset_ctor(root=cand_root, train=train, transform=transform, download=download)
                print("Using {} root: {}".format(dataset_name, cand_root))
                break
            except RuntimeError as e:
                last_err = e
                continue

        if base is None:
            tried = "\n".join(candidate_roots)
            raise RuntimeError(
                "Dataset not found. Tried roots:\n{}\nOriginal error: {}".format(tried, str(last_err))
            )

        self.base = base
        self.targets = base.targets
        print("Total # images:{}, classes:{}".format(len(self.base), self.num_classes))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, class_id = self.base[index]
        onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        onehot[int(class_id)] = 1.0
        item_name = "{}_{}".format("img", index)
        return item_name, index, image, onehot
