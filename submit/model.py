import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Architecture
        # number of discrimiator filters
        ndf = 64
        # number of colors
        nc = 3

        self.main = Discriminator(nc, ndf)

        # chop off the last two layers
        self.main = self.main.main[:-2]
        # define the size of the last layer now
        self.ninfeature = ndf * 8 * 4 * 4
        self.cls_hid_dim = 2048
        self.n_classes = 1000


        self.final = nn.Sequential(
            nn.Linear(self.ninfeature, self.cls_hid_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(self.cls_hid_dim, self.n_classes),
            nn.LogSoftmax(dim=1),
        )

        # Load pre-trained model
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        # TODO
        fvector = self.main(x)
        fvector = fvector.view(-1, self.ninfeature)
        logits = self.final(fvector)
        return (logits)
