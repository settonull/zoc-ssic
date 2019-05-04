from torchvision import datasets
import torch
import numpy as np
from torch.utils.data import Subset


def neg1(id):
    return -1

def image_loader(path, batch_size, transform, stype=None, eval_pct=1.0):

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

    elif 'unsupervised' in stype:
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



