import torch.nn as nn
import torchvision.transforms as transforms
import torch


h_size = 4096
z_size = 256

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=h_size):
        return input.view(input.size(0), size, 1, 1)


class Encoder(nn.Module):
    def __init__(self, image_channels=3, h_dim=h_size, z_dim=z_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten()
        )
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):

        if not self.training:
            return mu

        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        if torch.cuda.is_available():
            esp = esp.cuda()

        z = mu + std * esp
        return z

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    # If our particular model needs to do anything special to transform the image we can specify it here
    # will need to copy this over to eval.py if we do anything special
    transform = transforms.Compose(
        [
            #transforms.Resize((64, 64)),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261)
        ]
    )


class Decoder(nn.Module):
    def __init__(self, image_channels=3, h_dim=h_size, z_dim=z_size):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(z_dim, h_dim)

        #5,5,6,6
        #6,8,10,10
        #Stride 3 3 3 2
        #Kernal 4 5 6 8

        # Stride 1 2 3 3
        # Kernal 4 4 4 6

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=8, stride=2),
            nn.Sigmoid(),
        )

    def decode(self, z):
        z = self.fc(z)
        x = self.decoder(z)
        return x

    def forward(self, z):
        x = self.decode(z)
        return x




