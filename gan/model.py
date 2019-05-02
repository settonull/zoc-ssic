import torch
import torch.nn as nn
#from torch import optim
from tqdm import tqdm
from argparse import ArgumentParser
from models import dcgan
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

class Model(nn.Module):
    def __init__(self, batchsz = 64, mtype = 'df'):
        super(Model, self).__init__()
        # Architecture
        # TODO
        ndf = int(64);
        nc = int(3);
        ngpu = int(1);
        
        device = torch.device("cuda:0" if  torch.cuda.is_available()  else "cpu")
        self.encoder = dcgan.Discriminator(ngpu, nc, ndf).to(device);
        self.encoder = self.encoder.main[:-2];
        self.ninfeature = ndf*8*4*4;
        self.nhidden = 32;
        self.batchsz = batchsz;
        if mtype == 'df':
               self.final = nn.Sequential(
               nn.Linear(self.ninfeature, 1000),
               nn.LogSoftmax(dim=1),
               );
        # TODO
        # Load pre-trained model
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        if not torch.cuda.is_available():
         cuda = False;
        print('cuda', cuda);
        
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
        #raise NotImplementedError
        fvector = self.encoder(x);
        fvector = fvector.view(self.batchsz, self.ninfeature);
        y = self.final(fvector);
        return(y);

