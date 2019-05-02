import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from argparse import ArgumentParser
from models import dcgan
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

class classifier(nn.Module):
	def __init__(self, encoder, ninfeature, batchsz=10, mtype='df'):
		super(classifier, self).__init__();
		self.encoder = encoder;
		self.ninfeature = ninfeature;
		self.nhidden = 32;
		self.batchsz = batchsz;
		if mtype == 'df':
			self.final = nn.Sequential(
			nn.Linear(ninfeature, 1000),
			nn.LogSoftmax(dim=1),
			);

	def forward(self, x):
		fvector = self.encoder(x);
		fvector = fvector.view(self.batchsz, self.ninfeature);
		y = self.final(fvector);
		return(y);

def train(model, loader, lr, criterion = nn.NLLLoss(), optimizerclass = optim.SGD, gpu=True):
	if gpu:
		model = model.cuda();

	train_loss = 0;
	correct = 0;
	samples_processed = 0;
	model.train();
	optimizer = optimizerclass(model.parameters(), lr = lr);
	for batch in tqdm(loader):
		
		images = batch[0];
		labels = batch[1];
		
		batchsz = labels.size(0);
		
		if gpu:
			images = images.cuda();
			labels = labels.cuda();
		
		model.zero_grad();
		yhat = model(images);

		loss = criterion(yhat, labels);
		
		loss.backward();
		optimizer.step();
		
		prediction = torch.argmax(yhat, dim=1);
		correct += torch.sum(prediction==labels).item();
		samples_processed += batchsz;
		
		train_loss += loss.item() * batchsz;
		print('current train loss= ', loss.item());
		
	train_loss /= samples_processed;
	acc = correct/samples_processed;
	print('current train acc= ', acc);

	return model, train_loss, acc; 

def eval(model, loader, criterion = nn.NLLLoss(), gpu=True):
	if gpu:
		model = model.cuda();
	val_loss = 0;
	correct = 0;
	samples_processed = 0;
	model.eval();
	with torch.no_grad():
		for batch in tqdm(loader):
			
			images = batch[0];
			labels = batch[1];
			
			batchsz = labels.size(0);
			
			if gpu:
				images = images.cuda();
				labels = labels.cuda();
			
			yhat = model(images);
			
			loss = criterion(yhat, labels);
			
			#calculate the acc
			prediction = torch.argmax(yhat, dim=1);
			correct += torch.sum(prediction==labels).item();
			samples_processed += batchsz;
			
			val_loss += loss.item() * batchsz;
		
	val_loss /= samples_processed;
	acc = correct/samples_processed;
	print('\n|current val loss|current val acc |');
	print('+----------------+----------------+');
	print('|%5.2f           |%5.4f          |' % (loss.item(), acc));
	return val_loss, acc; 

def image_loader(path, batch_size, eval_pct=0.01):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)


    #just evaluate some of val
    indexes = [x for x in range(len(sup_val_data))]
    np.random.shuffle(indexes)
    amt = int(np.ceil(len(indexes) * eval_pct))
    subset_sup_val_data = Subset(sup_val_data, indexes[:amt])


    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        #subset_sup_val_data,
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup

def main():

	ap = ArgumentParser()
	ap.add_argument("-mt", "--model_type", default='mc-bert',
	                help="Name of model to use.")
	ap.add_argument("-nc", "--n_classes", type=int, default=3000,
	                help="Number of classes to predict.")
	ap.add_argument("-bs", "--batch_size", type=int, default=10,
	                help="Batch size for optimization.")
	ap.add_argument("-lr", "--learning_rate", type=float, default=3e-5,
	                help="Learning rate for optimization.")
	ap.add_argument("-ne", "--num_epochs", type=int, default=6,
	                help="Number of epochs for optimization.")
	ap.add_argument("-pt", "--patience", type=int, default=10,
	                help="Number of to wait before reducing learning rate.")
	ap.add_argument("-ml", "--min_lr", type=float, default=0.0,
	                help="Minimum learning rate.")
	ap.add_argument("-td", "--data_path",
	                help="Location of images.", default = '/scratch/zh1115/dsga1008/ssl_data_96/')
	ap.add_argument("-sd", "--save_dir",
	                help="Location to save the model.")
	ap.add_argument("-wd", "--weight_decay", type=float, default=1e-6,
	                help="Weight decay for nonbert models.")
	ap.add_argument("-ep", "--encoder_path",
			help="the path to load trained encoder(.pth file)", default = '/scratch/zh1115/dsga1008/gan_trained/netD_epoch_7.pth');
	ap.add_argument("-sp", "--save_path",
			help="the path to save the trained classifier", default = './gan_based_model_trained/');
	
	args = vars(ap.parse_args())
	ndf = int(64);
	nc = int(3);
	ngpu = int(1);
	if torch.cuda.is_available():
		if_gpu = True;
	else:
		if_gpu = False;
	
	device = torch.device("cuda:0" if  torch.cuda.is_available()  else "cpu")
	netD = dcgan.Discriminator(ngpu, nc, ndf).to(device)
	
	if not if_gpu:
		netD.load_state_dict(torch.load(args['encoder_path'], map_location = 'cpu')); #when cpu only
	else:
		netD.load_state_dict(torch.load(args['encoder_path']));
	#netD = torch.load(args['encoder_path'], map_location = 'cpu');
	
	ninfeature = ndf*8*4*4;
	clsfer = classifier(encoder = netD.main[:-2], ninfeature = ninfeature, batchsz = args['batch_size']);
	
	data_loader_sup_train, data_loader_sup_val, data_loader_unsup = image_loader(args['data_path'], batch_size=args['batch_size'])	
	
	eval(model=clsfer, loader=data_loader_sup_val, gpu = if_gpu);
	for i in range(args['num_epochs']):	
		clsfer, trainl, trainacc = train(model = clsfer, loader = data_loader_sup_train, lr = args['learning_rate'], criterion = nn.NLLLoss(), optimizerclass = optim.SGD, gpu=if_gpu);
		torch.save(clsfer.state_dict(), '%s/classifier_epoch_%d.pth' % (args['save_path'], i))	
		eval(model=clsfer, loader=data_loader_sup_val, gpu = if_gpu);
	
	print('------------ The end of training --------------')

if __name__ == "__main__":
	main();	
	
