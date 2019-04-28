import torch
import torch.nn as nn

from torchvision import models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.n_classes = 1000

        self.main = models.resnet18(pretrained=False)

        self.main.fc = nn.Sequential(nn.Linear(512, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(128, self.n_classes),
                                     nn.LogSoftmax(dim=1))
        # Load pre-trained model
        self.load_weights('weights.pth')

        #Seems to need this, at least on windows
        self.main = self.main.to('cuda')

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
        return self.main(x)
