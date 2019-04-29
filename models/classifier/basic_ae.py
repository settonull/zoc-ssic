import torch.nn as nn
from models.pretrain.basic_autoencoder import Encoder

class BasicAEClassifier(nn.Module):

    def __init__(self, n_classes=1000):
        super(BasicAEClassifier,self).__init__()

        self.encoder = Encoder()

        self.conv = nn.Conv2d(16, 1, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool2d((1,512))
        self.fc = nn.Linear(512, n_classes)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self ,x):
        # bs x 16 x 88 x 88
        x = self.encoder(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(1).squeeze(1)
        x = self.fc(x)
        x = self.lsm(x)
        return x

    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = Encoder.transform
