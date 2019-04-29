import torch.nn as nn
import torchvision.transforms as transforms


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder ,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6 ,16 ,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(True))

    def forward(self ,x):
        x = self.encoder(x)
        #bs x 16 x 88 x 88
        return x

    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = transforms.Compose(
        [
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
        ]
    )


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16 ,6 ,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6 ,3 ,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self ,x):
        x = self.decoder(x)
        return x

    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = transforms.Compose(
        [
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
        ]
    )
