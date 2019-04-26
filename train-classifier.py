"""Script for training Source Separation Unet."""

from argparse import ArgumentParser

from classifier_trainer import ClassifierTrainer

import torchvision.transforms as transforms
from torchvision import datasets
import torch


def image_loader(path, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)

    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


if __name__ == '__main__':
    """
    Usage:
        python train_classifier.py \
            --model_type basic
            --data_path /scratch/zh1115/dsga1008/ssl_data_96 \
            --save_dir saved_models
    """

    ap = ArgumentParser()
    ap.add_argument("-mt", "--model_type", default='mc-bert',
                    help="Name of model to use.")
    ap.add_argument("-nc", "--n_classes", type=int, default=3000,
                    help="Number of classes to predict.")
    ap.add_argument("-bs", "--batch_size", type=int, default=2,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=3e-5,
                    help="Learning rate for optimization.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=5,
                    help="Number of epochs for optimization.")
    ap.add_argument("-pt", "--patience", type=int, default=10,
                    help="Number of to wait before reducing learning rate.")
    ap.add_argument("-ml", "--min_lr", type=float, default=0.0,
                    help="Minimum learning rate.")
    ap.add_argument("-td", "--data_path",
                    help="Location of images.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=1e-6,
                    help="Weight decay for nonbert models.")

    args = vars(ap.parse_args())

    data_loader_sup_train, data_loader_sup_val, data_loader_unsup = image_loader(args['data_path'], batch_size=args['batch_size'])

    trainer = ClassifierTrainer(model_type=args['model_type'],
                     n_classes=args['n_classes'],
                     batch_size=args['batch_size'],
                     learning_rate=args['learning_rate'],
                     num_epochs=args['num_epochs'],
                     weight_decay=args['weight_decay'],
                     patience=args['patience'],
                     min_lr=args['min_lr'],

                            )


    trainer.fit(data_loader_sup_train, data_loader_sup_val, args['save_dir'])
