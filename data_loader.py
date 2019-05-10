from torchvision import datasets
import torch
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Sampler
import random

class ClassSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, desired_num_per_class, shuffle=False):

        self.data_source = data_source
        self.desired_num_per_class = desired_num_per_class
        self.total_items = len(data_source)
        self.total_classes = len(data_source.classes) #assume data_source returns this
        self.shuffle = shuffle

        assert self.total_items >= self.desired_num_per_class * self.total_classes, \
            "Can not have more samples than exist in dataset"

    def __iter__(self):

        # this assumes data_source has classes in order originally
        idxs = []
        class_size = self.total_items / self.total_classes

        for classn in range(self.total_classes):
            for i in range(self.desired_num_per_class):
                idxs.append( int((classn * class_size) + i ))

        assert len(idxs) == self.__len__()

        if self.shuffle:
            random.shuffle(idxs)

        return iter(idxs)

    def __len__(self):
        return self.total_classes * self.desired_num_per_class

def neg1(id):
    return -1

def image_loader(path, batch_size, transform, stype=None, eval_pct=1.0, desired_class_samples=-1):

    np.random.seed(1234)
    torch.manual_seed(1234)

    if 'supervised' in stype:
        print("Loading supervised datasets...", flush=True, end='')
        sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
        sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)

        # just evaluate some of val
        indexes = [x for x in range(len(sup_val_data))]
        np.random.shuffle(indexes)
        amt = int(np.ceil(len(indexes) * eval_pct))
        subset_sup_val_data = Subset(sup_val_data, indexes[:amt])

        if desired_class_samples != -1:
            smplr = ClassSampler(sup_train_data, desired_class_samples, shuffle=True)
            data_loader_sup_train = torch.utils.data.DataLoader(
                sup_train_data,
                batch_size=batch_size,
                #shuffle=True,
                num_workers=8,
                sampler=smplr
            )

        else:
            data_loader_sup_train = torch.utils.data.DataLoader(
                sup_train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8
            )
        data_loader_sup_val = torch.utils.data.DataLoader(
            subset_sup_val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        print("done.", flush=True)
        return data_loader_sup_train, data_loader_sup_val

    elif 'un' in stype:
        print("Loading unsupervised datasets...", flush=True, end='')
        unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
        data_loader_unsup = torch.utils.data.DataLoader(
            unsup_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        print("done.", flush=True)
        return data_loader_unsup

    elif 'semi' in stype:
        print("Loading semi-supervised datasets...", flush=True, end='')
        sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
        sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
        unsup_train_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform, target_transform=neg1)

        semi_train_data = torch.utils.data.ConcatDataset([sup_train_data, unsup_train_data ])

        # just evaluate some of val
        indexes = [x for x in range(len(sup_val_data))]
        np.random.shuffle(indexes)
        amt = int(np.ceil(len(indexes) * eval_pct))
        subset_sup_val_data = Subset(sup_val_data, indexes[:amt])

        data_loader_sup_train = torch.utils.data.DataLoader(
            semi_train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        data_loader_sup_val = torch.utils.data.DataLoader(
            subset_sup_val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        print("done.", flush=True)
        return data_loader_sup_train, data_loader_sup_val



