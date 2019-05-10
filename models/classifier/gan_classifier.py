import torch
from torch import nn

from models.pretrain.dcgan import Discriminator


class GANClassifier(nn.Module):

    def __init__(self, n_classes=1000, cls_hid_dim=4000, extra_cls=False):
        super(GANClassifier,self).__init__()

        # number of discrimiator filters
        ndf = 64
        # number of colors
        nc = 3

        self.cls_hid_dim = cls_hid_dim
        self.n_classes = n_classes
        self.main = Discriminator(nc, ndf)

        # chop off the last two layers
        self.main = self.main.main[:-2]
        #define the size of the last layer now
        self.ninfeature = ndf * 8 * 4 * 4

        if extra_cls:
            self.final = nn.Sequential(
                nn.Linear(self.ninfeature, self.ninfeature),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(self.ninfeature, self.cls_hid_dim),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(self.cls_hid_dim, self.n_classes),
                nn.LogSoftmax(dim=1),
            )
        else:
            self.final = nn.Sequential(
                nn.Linear(self.ninfeature, self.cls_hid_dim),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(self.cls_hid_dim, self.n_classes),
                nn.LogSoftmax(dim=1),
            )

    def forward(self, x):

        fvector = self.main(x)
        bs = fvector.shape[0]
        fvector = fvector.view(bs, self.ninfeature)
        logits = self.final(fvector)
        return (logits)


    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = Discriminator.transform

