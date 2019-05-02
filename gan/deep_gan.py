from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')

#%matplotlib inline
import matplotlib.pyplot as plt

from utils import image_loader
import numpy as np

dataroot = os.path.normpath('/scratch/zh1115/dsga1008/ssl_data_96/') #path to dataset
batchSize = 64 #input batch size
nz = 100 #size of the latent z vector
ngf = 64 #number of generator filters
ndf = 64 #number of discrimiator filters
niter = 25 #number of epochs to train for
lr = 0.0002 #learning rate
beta1 = 0.5 #beta1 for adam
netG_path = '' #path to netG (to continue training)
netD_path = '' #path to netD (to continue training)
outf = '/scratch/zh1115/dsga1008/gan_trained/' #folder to output images and model checkpoints
manualSeed = None #manual seed
cuda = True
ngpu = 1


try:
    os.makedirs(outf)
except OSError:
    print('OSError');
    pass;
except:
    print('Error while creating out put folder');
    raise;

if manualSeed is None:
    manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with cuda=True")


#DATA LOADER

#path = os.path.join(os.getcwd(), 'data\ssl_data_96')
path = dataroot;
_, _, dataloader = image_loader(path, batchSize)
imageSize = 96 #the height / width of the input image
nc=3

device = torch.device("cuda:0" if cuda else "cpu")
nz = int(nz)
ngf = int(ngf)
ndf = int(ndf)

image, _ = dataloader.dataset.__getitem__(0)
print('{} images in the data set'.format(len(dataloader.dataset)))

# show some training images
#plt.figure(figsize=(16, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    idx = np.random.randint(len(dataloader.dataset))
    image, _ = dataloader.dataset.__getitem__(idx)
    plt.imshow(image.permute(1,2,0).numpy())
    plt.axis('off')
plt.savefig('examplefromloader.png');

from models.dcgan import Generator, Discriminator, weights_init

netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)
if netG_path != '':
    netG.load_state_dict(torch.load(netG_path))
print(netG)

netD = Discriminator(ngpu, nc, ndf).to(device)
netD.apply(weights_init)
if netD_path != '':
    netD.load_state_dict(torch.load(netD_path))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


#Model training
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))


