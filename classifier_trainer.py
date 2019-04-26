
import os
#import csv
import random

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim import Adam

from basic_classifier import BasicClassifier

class ClassifierTrainer():

    """Class to train and evaluate BertVisualMemory."""

    def __init__(self, model_type='basic', n_classes=1000, batch_size=64,
                 learning_rate=3e-5, num_epochs=100, weight_decay=0,
                 patience=10, min_lr=0):
        """
        Initialize BertMBC model.

        Args
        ----
            model_type : string, model name, 'mc-bert'.
            vis_feat_dim : int, intermediate visual feature dimension.
            spatial_size : int, spatial size of visual features.
            lm_hidden_dim : int, size of hidden state in language model.
            cmb_feat_dim : int, combined feature dimension.
            kernel_size : int, kernel_size to use in attention.
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

        # Model attributes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.nn_epoch = 0
        self.best_val_acc = 0
        self.save_dir = None

        # reproducability attributes
        self.torch_rng_state = None
        self.numpy_rng_state = None

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self):
        """Initialize the nn model for training."""
        if self.model_type == 'basic':
            mode = BasicClassifier(self.n_classes)
        else:
            raise ValueError("Did not recognize model type!")

        self.optimizer = Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', verbose=True, patience=self.patience,
            min_lr=self.min_lr)

        if self.USE_CUDA:
            self.model = self.model.cuda()

        # reproducability and deteriministic continuation of models
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.torch_rng_state = torch.get_rng_state()
        self.numpy_rng_state = np.random.get_state()

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
            logits = self.model(images)

            probs = torch.nn.functional.log_softmax(logits, dim=1)
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
                labels = batch_samples['labels']

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

    def fit(self, train_loader, val_loader, save_dir, warm_start=False):
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
               Weight Decay: {}\n\
               Learning Rate: {}\n\
               Patience: {}\n\
               Min LR: {}\n\
               Batch Size: {}\n\
               N Classes: {}\n\
               Save Dir: {}".format(
                   self.model_type, self.weight_decay, self.learning_rate,
                   self.patience, self.min_lr, self.batch_size, train_chunks, eval_pct,
                   self.n_classes, save_dir), flush=True)

        self.save_dir = save_dir
        self.model_dir = self._format_model_subdir()

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn()
        train_loss = 0
        train_acc = 0

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
                self.best_val_acc = val_acc
                self.torch_rng_state = torch.get_rng_state()
                self.numpy_rng_state = np.random.get_state()
                self.save()
            self.nn_epoch += 1

            if self.scheduler:
                self.scheduler.step(val_acc)

    def score(self, loader):
        """
        Score all predictions.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        raise NotImplementedError("Not yet implemented!")

    def predict(self, loader):
        """
        Predict for an input.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        raise NotImplementedError("Not yet implemented!")

    '''
    def report_results(self, val_dataset, outfile_name):

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        f = open(outfile_name, 'w', newline='')
        writer = csv.writer(f)

        val_loss, val_acc = self._eval_epoch(val_loader, (writer, f))
        f.close()

        print("\nVal Loss: {}\tVal Acc: {}".format(
            np.round(val_loss, 5), np.round(val_acc * 100, 2)), flush=True)
    '''

    def _format_model_subdir(self):
        subdir = "classifier_mt{}-lr{}-nc{}-wd{}-pt{}-mlr".\
                format(self.model_type, self.learning_rate,
                       self.n_classes,
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
        if (self.model is not None) and (self.save_dir is not None):

            if not os.path.isdir(os.path.join(self.save_dir, self.model_dir)):
                os.makedirs(os.path.join(self.save_dir, self.model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(self.save_dir, self.model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'trainer_dict': self.__dict__}, file)

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
