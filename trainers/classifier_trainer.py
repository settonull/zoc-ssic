
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim import SGD

from models.classifier.basic_classifier import BasicClassifier
from models.classifier.basic_ae import BasicAEClassifier
from models.classifier.vae_z_classifier import VAEZClassifier
from models.classifier.vae_h_classifier import VAEHClassifier
from models.classifier.gan_classifier import GANClassifier

from data_loader import image_loader


class ClassifierTrainer():

    """Class to train and evaluate BertVisualMemory."""

    def __init__(self, model_type='basic', n_classes=1000, batch_size=64,
                 learning_rate=3e-5, num_epochs=100, weight_decay=0,
                 patience=10, min_lr=0, eval_pct=0.05, pretrain_weights=None, cls_hid_dim=2048,
                 remove_previous=True, early_stopping=5, extra_cls=False, opt='adam', desired_class_samples=-1):
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
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_lr = min_lr
        self.eval_pct = eval_pct
        self.remove_previous = remove_previous
        self.last_save_file = None
        self.early_stopping = early_stopping
        self.opt = opt.lower()

        # Model attributes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.nn_epoch = 0
        self.best_val_acc = 0
        self.save_dir = None
        self.pretrain_weights = pretrain_weights
        self.cls_hid_dim = cls_hid_dim
        self.extra_cls = extra_cls
        self.desired_class_samples = desired_class_samples

        # reproducability attributes
        self.torch_rng_state = None
        self.numpy_rng_state = None

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self):
        """Initialize the nn model for training."""
        if self.model_type == 'basic':
            self.model = BasicClassifier(self.n_classes)
        elif self.model_type == 'ae-basic':
            self.model = BasicAEClassifier(self.n_classes)
            if self.pretrain_weights:
                self.model.encoder.load_state_dict(torch.load(self.pretrain_weights)['state_dict_1'])

        elif self.model_type == 'vae-h':
            self.model = VAEHClassifier(self.n_classes, self.cls_hid_dim)
            if self.pretrain_weights:
                self.model.encoder.load_state_dict(torch.load(self.pretrain_weights)['state_dict_1'])

        elif self.model_type == 'vae-z':
            self.model = VAEZClassifier(self.n_classes)
            if self.pretrain_weights:
                self.model.encoder.load_state_dict(torch.load(self.pretrain_weights)['state_dict_1'])
        elif self.model_type == 'gan':
            self.model = GANClassifier(self.n_classes, self.cls_hid_dim, extra_cls=self.extra_cls)
            if self.pretrain_weights:
                w = torch.load(self.pretrain_weights)
                if 'state_dict_2' in w:
                    self.model.load_state_dict(w['state_dict_2'], strict=False)
                else:
                    self.model.load_state_dict(w, strict=False)

        else:
            raise ValueError("Did not recognize model type!")

        if self.opt == 'adam':
            self.optimizer = Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)
        elif self.opt == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(), lr=self.learning_rate)
        else:
            print("Unknown optimizer", self.opt, "only adam or sgd supported")
            exit(-1)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', verbose=True, patience=self.patience,
            min_lr=self.min_lr)

        if self.USE_CUDA:
            self.model = self.model.cuda()

        # reproducability and deteriministic continuation of models
        np.random.seed(1234)
        torch.manual_seed(1234)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #self.torch_rng_state = torch.get_rng_state()
        #self.numpy_rng_state = np.random.get_state()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        correct = 0
        loss_fct = torch.nn.NLLLoss()

        for batch_samples in tqdm(loader):

            # prepare training sample
            # batch_size x RGB x height X width
            images = batch_samples[0]
            # batch_size
            labels = batch_samples[1]

            #could be smaller due to end of loader
            batch_size = labels.size(0)

            if self.USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()

            # forward pass
            self.model.zero_grad()
            #let's calculate loss and accuracy out here
            probs = self.model(images)

            loss = loss_fct(probs, labels)

            # backward pass
            loss.backward()
            self.optimizer.step()

            # compute train loss and acc
            predicts = torch.argmax(probs, dim=1)
            correct += torch.sum(predicts == labels).item()

            samples_processed += batch_size
            train_loss += loss.item() * batch_size

        train_loss /= samples_processed
        acc = correct / samples_processed

        return train_loss, acc

    def _eval_epoch(self, loader, outfile=None):
        """Eval epoch."""
        self.model.eval()
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
                    images = images.cuda()
                    labels = labels.cuda()

                # forward pass
                # let's calculate loss and accuracy out here
                logits = self.model(images)
                probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = loss_fct(probs, labels)

                # compute train loss and acc
                predicts = torch.argmax(probs, dim=1)
                correct += torch.sum(predicts == labels).item()

                #write out results
                #if outfile is not None:
                #    qids = batch_samples['qids']
                #    for i in range(len(qids)):
                #        outfile[0].writerow([qids[i].item(), predicts[i].item(), labels[i].item()])
                #    outfile[1].flush()


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
            train_dataset : PyTorch dataset, training data.
            val_dataset : PyTorch dataset, validation data.
            save_dir: directory to save nn_model

        """
        # Print settings to output file
        print("Settings:\n\
               Model Type: {}\n\
               Optimizer: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Patience: {}\n\
               Min LR: {}\n\
               Batch Size: {}\n\
               N Classes: {}\n\
               Classifier hidden: {}\n\
               Samples per Classes: {}\n\
               Extra Classifer Layer: {}\n\
               Save Dir: {}".format(
                   self.model_type, self.opt, self.learning_rate,
                   self.weight_decay, self.patience, self.min_lr, self.batch_size,
                   self.n_classes, self.cls_hid_dim, self.desired_class_samples,
                   self.extra_cls, save_dir), flush=True)

        self.save_dir = save_dir
        self.model_dir = self._format_model_subdir()

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn()
        train_loss = 0
        train_acc = 0

        best_epoch = 0

        train_loader, val_loader = image_loader(data_dir, batch_size=self.batch_size, transform=self.model.transform,
                                                stype='supervised', eval_pct=self.eval_pct, desired_class_samples=self.desired_class_samples)

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            if self.nn_epoch > 0:
                print("\nInitializing train epoch...", flush=True)
                train_loss, train_acc = self._train_epoch(train_loader)

            print("\nInitializing val epoch...", flush=True)
            val_loss, val_acc = self._eval_epoch(val_loader)

            # report
            print("\nEpoch: [{}/{}]\tTrain Loss: {}\tTrain Acc: {}\tVal Loss: {}\tVal Acc: {}".format(
                self.nn_epoch, self.num_epochs, np.round(train_loss, 5),np.round(train_acc * 100, 2),
                np.round(val_loss, 5), np.round(val_acc * 100, 2)), flush=True)

            # save best
            if val_acc > self.best_val_acc:
                best_epoch = self.nn_epoch
                self.best_val_acc = val_acc
                self.torch_rng_state = torch.get_rng_state()
                self.numpy_rng_state = np.random.get_state()
                self.save()

            if (self.early_stopping > 0) and ((self.nn_epoch - best_epoch) >= self.early_stopping):
                print("No progress in {} epochs, best acc: {}, exiting".format(self.nn_epoch, self.best_val_acc))
                break

            if self.scheduler:
                self.scheduler.step(val_acc)

            self.nn_epoch += 1

    def _format_model_subdir(self):
        subdir = "classifier_mt-{}_lr-{}_wd-{}_pt-{}_mlr-{}_hid-{}_opt-{}_extr-{}".\
                format(self.model_type, self.learning_rate,
                       self.weight_decay, self.patience,
                       self.min_lr, self.cls_hid_dim, self.opt, self.extra_cls)
        return subdir

    def save(self):
        """
        Save model.

        Args
        ----
            models_dir: path to directory for saving NN models.

        """
        if (self.model is not None) and (self.save_dir is not None):

            if not os.path.isdir(os.path.join(self.save_dir, self.model_dir)):
                os.makedirs(os.path.join(self.save_dir, self.model_dir))

            if self.remove_previous and self.last_save_file is not None:
                os.remove(self.last_save_file)

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(self.save_dir, self.model_dir, filename)
            self.last_save_file = fileloc
            with open(fileloc, 'wb') as file:
                torch.save(self.model.state_dict(), file)

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
