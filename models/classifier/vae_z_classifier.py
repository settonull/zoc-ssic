from torch import nn
from models.pretrain.vae import Encoder


class VAEZClassifier(nn.Module):

    def __init__(self, n_classes=1000):
        super(VAEZClassifier,self).__init__()

        self.encoder = Encoder()

        self.fc1 = nn.Linear(self.encoder.z_dim, 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, n_classes)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self ,x):
        # bs x 16 x 88 x 88
        z, mu, logvar = self.encoder(x)

        x = self.fc1(z)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.lsm(x)
        return x

    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = Encoder.transform


