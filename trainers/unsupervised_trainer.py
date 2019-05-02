import os
import numpy as np
from tqdm import tqdm
import math

import torch
from torch.optim import Adam

#for the GAN model
from models.pretrain.dcgan import Generator,Discriminator, weights_init
from models.pretrain.basic_autoencoder import Encoder, Decoder
from models.pretrain.vae import Encoder as VAE_Encoder, Decoder as VAE_Decoder

#module to load all our images
from data_loader import image_loader

from torchvision.utils import save_image

class UnsupervisedTrainer():

    """Class to train and evaluate BertVisualMemory."""

    def __init__(self, model_type='basic', batch_size=64,
                 learning_rate=0.0002, num_epochs=100, weight_decay=0,
                 patience=10, min_lr=0, eval_pct=0.05):
        """
        Initialize Classifier trainer.

        Args
        ----
            model_type : string, model name, 'basic'.
            batch_size : int, batch size for optimization.
            num_epochs : int, number of epochs to train for.

        """
        # Trainer attributes
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_lr = min_lr
        self.eval_pct = eval_pct

        # Model attributes
        self.model_1 = None
        self.model_2 = None
        self.optimizer_1 = None
        self.optimizer_2 = None
        self.scheduler_1 = None
        self.scheduler_2 = None
        self.nn_epoch = 1
        self.best_eval_criteria = 0
        self.save_dir = None

        # reproducability attributes
        self.torch_rng_state = None
        self.numpy_rng_state = None

        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.USE_CUDA else "cpu")

    def _init_nn(self):
        """Initialize the nn model for training."""

        #Initalize everything for our model
        if self.model_type == 'gan':
            #should make these passed in paramaters
            self.nz = 100  # size of the latent z vector
            self.ngf = 64  # number of generator filters
            self.ndf = 64  # number of discrimiator filters
            self.nc =3
            self.ngpu = 1
            self.imageSize = 96

            self.model_1 = Generator(self.ngpu, self.nz, self.ngf, self.nc)
            self.model_1.apply(weights_init)
            self.model_2 = Discriminator(self.ngpu, self.nc, self.ndf)
            self.model_2.apply(weights_init)

            self.fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
            self.real_label = 1
            self.fake_label = 0

        elif self.model_type == 'ae-basic':
            #should make these passed in paramaters

            self.model_1 = Encoder()
            self.model_2 = Decoder()

        elif self.model_type == 'vae':

            self.model_1 = VAE_Encoder()
            self.model_2 = VAE_Decoder()


        else:
            raise ValueError("Did not recognize model type!")

        self.optimizer_1 = Adam(
            self.model_1.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        #self.scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, 'max', verbose=True, patience=self.patience,
        #    min_lr=self.min_lr)

        self.optimizer_2 = Adam(
            self.model_2.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        #self.scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, 'max', verbose=True, patience=self.patience,
        #    min_lr=self.min_lr)

        if self.USE_CUDA:
            self.model_1 = self.model_1.cuda()
            self.model_2 = self.model_2.cuda()

        # reproducability and deteriministic continuation of models
        #np.random.seed(1234)
        #torch.manual_seed(1234)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #self.torch_rng_state = torch.get_rng_state()
        #self.numpy_rng_state = np.random.get_state()

    #since an unsupervised training loop can be very custom, we just define our own here
    def _train_GAN_epoch(self, loader):
        """Train epoch."""
        self.model_1.train()
        self.model_2.train()

        loss_fct = torch.nn.BCELoss()

        #using specifc names just to make the code clearer
        netG = self.model_1
        optimizerG = self.optimizer_1
        netD = self.model_2
        optimizerD = self.optimizer_2

        total_errD = 0
        total_errG = 0
        total_D_x = 0
        total_D_G_z1 = 0
        total_D_G_z2 = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # train with real
            netD.zero_grad()
            real_cpu = batch_samples[0].to(self.device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), self.real_label, device=self.device)
            output = netD(real_cpu)
            errD_real = loss_fct(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake = netG(noise)
            label.fill_(self.fake_label)
            output = netD(fake.detach())
            errD_fake = loss_fct(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = loss_fct(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            samples_processed += batch_size
            total_errG += errG.item() * batch_size
            total_errD += errD.item() * batch_size
            total_D_x += D_x * batch_size
            total_D_G_z1 += D_G_z1 * batch_size
            total_D_G_z2 += D_G_z2 * batch_size

        total_errG /= samples_processed
        total_errD /= samples_processed
        total_D_x /= samples_processed
        total_D_G_z1 /= samples_processed
        total_D_G_z2 /= samples_processed

        report = 'Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f} '.format(
                 total_errD, total_errG, total_D_x, total_D_G_z1, total_D_G_z2)

        return report, -total_errD, None

    # since an unsupervised training loop can be very custom, we just define our own here
    def _train_AE_BASIC_epoch(self, loader):
        """Train epoch."""
        self.model_1.train()
        self.model_2.train()

        loss_fct = torch.nn.MSELoss()

        total_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):
            # train with real
            self.model_1.zero_grad()
            self.model_2.zero_grad()

            img = batch_samples[0].to(self.device)
            batch_size = img.shape[0]

            enc = self.model_1(img)
            output = self.model_2(enc)

            loss = loss_fct(output, img)
            loss.backward()
            self.optimizer_1.step()
            self.optimizer_2.step()

            samples_processed += batch_size
            total_loss += loss.item() * batch_size

        total_loss /= samples_processed

        report = 'Loss: {:.4f} '.format(total_loss)

        return report, -total_loss, None

        # since an unsupervised training loop can be very custom, we just define our own here
    def _train_VAE_epoch(self, loader):
        """Train epoch."""
        self.model_1.train()
        self.model_2.train()

        total_loss = 0
        total_bce = 0
        total_kld = 0
        samples_processed = 0

        def loss_fn(recon_x, x, mu, logvar):
            BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
            # BCE = F.mse_loss(recon_x, x, size_average=False)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            return BCE + KLD, BCE, KLD

        for batch_samples in tqdm(loader):
            # train with real
            self.model_1.zero_grad()
            self.model_2.zero_grad()

            img = batch_samples[0].to(self.device)
            batch_size = img.shape[0]

            z, mu, logvar = self.model_1(img)
            output = self.model_2(z)

            #print("in:",img.shape)
            #print("out:", output.shape)

            loss, bce, kld = loss_fn(output, img, mu, logvar)
            loss.backward()
            self.optimizer_1.step()
            self.optimizer_2.step()

            samples_processed += batch_size
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        test_imgs = []
        for i in range(int(math.ceil(batch_size/8))):
            test_imgs.append(torch.cat([img[i], output[i]], dim=2))

        test_img = torch.cat(test_imgs, dim=1)

        total_loss /= samples_processed
        total_bce /= samples_processed
        total_kld /= samples_processed

        report = "Loss: {:.3f} {:.3f} {:.3f}".format(total_loss, total_bce, total_kld)

        return report, -total_loss, test_img

    def fit(self, data_dir, save_dir, warm_start=False):
        """
        Train the NN model.

        Args
        ----
            data_dir : Where our data is
            save_dir: directory to save nn_model

        """
        # Print settings to output file
        print("Settings:\n\
               Model Type: {}\n\
               Weight Decay: {}\n\
               Learning Rate: {}\n\
               Patience: {}\n\
               Min LR: {}\n\
               Batch Size: {}\n\
               Save Dir: {}".format(
                   self.model_type, self.weight_decay, self.learning_rate,
                   self.patience, self.min_lr, self.batch_size,
                    save_dir), flush=True)

        self.save_dir = save_dir
        self.model_dir = self._format_model_subdir()

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn()

        train_loader = image_loader(data_dir, batch_size=self.batch_size, transform=self.model_1.transform,
                                        supervised=False, eval_pct=self.eval_pct)

        #specify our train method here, as well as intialize what a "good" criteria is
        if self.model_type =='gan':
            train_epoch = self._train_GAN_epoch
            #use negative since we want larger numbers (but smaller loss) to be better performance
            self.best_eval_criteria = -100000

        if self.model_type =='ae-basic':
            train_epoch = self._train_AE_BASIC_epoch
            #use negative since we want larger numbers (but smaller loss) to be better performance
            self.best_eval_criteria = -100000

        if self.model_type =='vae':
            train_epoch = self._train_VAE_epoch
            #use negative since we want larger numbers (but smaller loss) to be better performance
            self.best_eval_criteria = -100000


        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            print("\nInitializing train epoch...", flush=True)
            report, eval_criteria, timg = train_epoch(train_loader)

            if(timg is not None):
                save_image(timg.cpu(), 'latest-' + str(self.nn_epoch) + '.png')

            # report
            print("\nEpoch: [{}/{}]".format(self.nn_epoch, self.num_epochs) + report, flush=True)

            # save best
            if eval_criteria > self.best_eval_criteria:
                self.best_val_acc = eval_criteria
                self.torch_rng_state = torch.get_rng_state()
                self.numpy_rng_state = np.random.get_state()
                self.save()
            self.nn_epoch += 1

            if self.scheduler_1:
                self.scheduler.step(eval_criteria)

            if self.scheduler_2:
                self.scheduler.step(eval_criteria)


    def _format_model_subdir(self):
        subdir = "unsupervised_mt{}-lr{}-wd{}-pt{}-mlr".\
                format(self.model_type, self.learning_rate,
                       self.weight_decay, self.patience,
                       self.min_lr)
        return subdir

    def save(self):
        """
        Save model.

        Args
        ----
            models_dir: path to directory for saving NN models.

        """
        if (self.model_1 is not None) and (self.save_dir is not None):

            if not os.path.isdir(os.path.join(self.save_dir, self.model_dir)):
                os.makedirs(os.path.join(self.save_dir, self.model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(self.save_dir, self.model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict_1': self.model_1.state_dict(),
                            'state_dict_2': self.model_2.state_dict()
                            }, file)

    def load(self, model_dir, epoch, train_chunks=0, train_data_len=None):
        """
        Load a previously trained model.

        Args
        ----
            model_dir : directory where models are saved.
            epoch : epoch of model to load.

        """

        skip_list = []

        epoch_file = "epoch_{}".format(epoch) + '.pth'
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')

        for (k, v) in checkpoint['trainer_dict'].items():
            if k not in skip_list:
                setattr(self, k, v)

        self.USE_CUDA = torch.cuda.is_available()
        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
        torch.set_rng_state(self.torch_rng_state)
        np.random.set_state(self.numpy_rng_state)
        self.nn_epoch += 1
