import os
import numpy as np
from tqdm import tqdm
import math

import torch
from torch.optim import Adam

from models.ssl.vae import Encoder as VAE_Encoder, Decoder as VAE_Decoder, VAEHClassifier

#module to load all our images
from data_loader import image_loader

from torchvision.utils import save_image

class SemiSupervisedTrainer():

    """Class to train and evaluate BertVisualMemory."""

    def __init__(self, model_type='basic',n_classes=1000, batch_size=64,
                 learning_rate=0.0002, num_epochs=100, weight_decay=0,
                 patience=10, min_lr=0, eval_pct=0.05, cl_pct=0.5):
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
        self.n_classes=n_classes

        # Model attributes
        self.model_1 = None
        self.model_2 = None
        self.model_3 = None
        self.optimizer_1 = None
        self.scheduler_1 = None
        self.nn_epoch = 0
        self.best_eval_criteria = 0
        self.save_dir = None
        self.cl_pct = cl_pct

        # reproducability attributes
        self.torch_rng_state = None
        self.numpy_rng_state = None

        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.USE_CUDA else "cpu")

    def _init_nn(self):
        """Initialize the nn model for training."""

        if self.model_type == 'vae':

            self.model_1 = VAE_Encoder()
            self.model_2 = VAE_Decoder()
            self.model_3 = VAEHClassifier(self.n_classes)

        else:
            raise ValueError("Did not recognize model type!")

        params = list(self.model_1.parameters()) + list(self.model_2.parameters())+ list(self.model_3.parameters())

        self.optimizer_1 = Adam(params, lr=self.learning_rate,
            weight_decay=self.weight_decay)

        #self.scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, 'max', verbose=True, patience=self.patience,
        #    min_lr=self.min_lr)

        #self.optimizer_2 = Adam(
        #    self.model_2.parameters(), lr=self.learning_rate,
        #    weight_decay=self.weight_decay)

        #self.scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, 'max', verbose=True, patience=self.patience,
        #    min_lr=self.min_lr)

        if self.USE_CUDA:
            self.model_1 = self.model_1.to(self.device)
            self.model_2 = self.model_2.to(self.device)
            self.model_3 = self.model_3.to(self.device)

        # reproducability and deteriministic continuation of models
        #np.random.seed(1234)
        #torch.manual_seed(1234)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #self.torch_rng_state = torch.get_rng_state()
        #self.numpy_rng_state = np.random.get_state()


        # since an unsupervised training loop can be very custom, we just define our own here
    def _train_VAE_epoch(self, loader):
        """Train epoch."""
        self.model_1.train()
        self.model_2.train()
        self.model_3.train()

        total_loss = 0
        total_bce = 0
        total_kld = 0
        samples_processed = 0
        total_predict_loss = 0
        lbl_samples_processed = 0
        correct = 0

        cl_pct = self.cl_pct

        prd_loss_fct = torch.nn.NLLLoss(ignore_index=-1)

        def rec_loss_fct(recon_x, x, mu, logvar):
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
            self.model_3.zero_grad()

            img = batch_samples[0].to(self.device)
            labels = batch_samples[1].to(self.device)

            batch_size = img.shape[0]

            #reconstruct image
            z, mu, logvar, h = self.model_1(img)
            output = self.model_2(z)

            #make classifications
            logits = self.model_3(h)
            probs = torch.nn.functional.log_softmax(logits, dim=1)

            recon_loss, bce, kld = rec_loss_fct(output, img, mu, logvar)
            #print("porbs:", probs.shape, "labels:", labels.shape, labels)
            predict_loss = prd_loss_fct(probs, labels)

            #mask out the loss from any unlabeled
            mask = labels != -1
            lbl_smpls = mask.sum().item()
            #predict_loss_list *= mask.float()
            #predict_loss = predict_loss_list.sum() / lbl_smpls

            # compute acc
            predicts = torch.argmax(probs, dim=1)
            correct += torch.sum(predicts == labels).item()

            #add the losses
            loss = (recon_loss * (1-cl_pct)) + (predict_loss * cl_pct)
            loss.backward()

            self.optimizer_1.step()

            samples_processed += batch_size
            lbl_samples_processed += lbl_smpls

            total_predict_loss += predict_loss.item() * lbl_smpls
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        test_imgs = []
        for i in range(int(math.ceil(batch_size/8))):
            test_imgs.append(torch.cat([img[i], output[i]], dim=2))

        test_img = torch.cat(test_imgs, dim=1)

        total_predict_loss /= lbl_samples_processed
        total_loss /= samples_processed
        total_bce /= samples_processed
        total_kld /= samples_processed
        acc = (correct / lbl_samples_processed) * 100

        report = "Loss: {:.3f} {:.3f} {:.3f} {:.3f}\tAcc:{:.2f}".format(
            total_predict_loss, total_loss, total_bce, total_kld, acc)

        return report, total_loss, test_img

    def _eval_epoch(self, loader, outfile=None):
        """Eval epoch."""
        self.model_1.eval()
        self.model_3.eval()
        val_loss = 0
        samples_processed = 0
        correct = 0
        loss_fct = torch.nn.NLLLoss()

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                # batch_size x RGB x height X width
                images = batch_samples[0]
                # batch_size
                labels = batch_samples[1]

                batch_size = images.size(0)

                if self.USE_CUDA:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                # forward pass
                # let's calculate loss and accuracy out here
                _, _, _, h = self.model_1(images)
                logits = self.model_3(h)

                probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = loss_fct(probs, labels)

                # compute train loss and acc
                predicts = torch.argmax(probs, dim=1)
                correct += torch.sum(predicts == labels).item()

                samples_processed += batch_size
                val_loss += loss.item() * batch_size

            val_loss /= samples_processed
            acc = correct / samples_processed

        return val_loss, acc




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
               Classifer pct: {}\n\
               Weight Decay: {}\n\
               Learning Rate: {}\n\
               Patience: {}\n\
               Min LR: {}\n\
               Batch Size: {}\n\
               Save Dir: {}".format(
                   self.model_type, self.cl_pct, self.weight_decay, self.learning_rate,
                   self.patience, self.min_lr, self.batch_size,
                    save_dir), flush=True)

        self.save_dir = save_dir
        self.model_dir = self._format_model_subdir()

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn()

        train_loader, val_loader = image_loader(data_dir, batch_size=self.batch_size, transform=self.model_1.transform,
                                        stype='semi',  eval_pct=self.eval_pct)

        #specify our train method here, as well as intialize what a "good" criteria is
        if self.model_type =='vae':
            train_epoch = self._train_VAE_epoch
            #use negative since we want larger numbers (but smaller loss) to be better performance
            self.best_eval_criteria = 0


        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            if self.nn_epoch > 0:
                print("\nInitializing train epoch...", flush=True)
                report, eval_criteria, timg = train_epoch(train_loader)
            else:
                report = ""
                timg = None

            print("\nInitializing val epoch...", flush=True)
            val_loss, val_acc = self._eval_epoch(val_loader)

            eval_criteria = val_acc

            # report
            print("\nEpoch: [{}/{}]\t".format(self.nn_epoch, self.num_epochs)  +  report +
                  "\tVal Loss: {:.3f}\tVal Acc: {:.3f}".format(val_loss, val_acc * 100), flush=True)


            if(timg is not None):
                save_image(timg.cpu(), 'semi-latest-' + str(self.nn_epoch) + '.png')

            # save best
            if eval_criteria > self.best_eval_criteria:
                self.best_val_acc = eval_criteria
                self.torch_rng_state = torch.get_rng_state()
                self.numpy_rng_state = np.random.get_state()
                self.save()
            self.nn_epoch += 1

            if self.scheduler_1:
                self.scheduler.step(eval_criteria)


    def _format_model_subdir(self):
        subdir = "semi_mt-{}_lr-{}_wd-{}_pt-{}_mlr-{}_cp-{}".\
                format(self.model_type, self.learning_rate,
                       self.weight_decay, self.patience,
                       self.min_lr, self.cl_pct)
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
