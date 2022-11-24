import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import my_train_help

argp = argparse.ArgumentParser(description='Train.py')

argp.add_argument('data_dir', nargs='*', action="store", default="ImageClassifier/flowers")
argp.add_argument('--gpu', dest="gpu", action="store", default="cuda")
argp.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argp.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argp.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argp.add_argument('--epochs', dest="epochs", action="store", type=int, default=9)
argp.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
argp.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

pa = argp.parse_args()

loc = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
strct = pa.arch
dropout = pa.dropout
hidden_layer = pa.hidden_units
device = pa.gpu
epochs = pa.epochs


dataloaders ,datasizes,image_datasets= my_train_help.data_loading(loc)

model,criteria , optimizer , sched = my_train_help.model_setup(strct  ,dropout,hidden_layer,lr,device)

model = my_train_help.train_model(model, criteria, optimizer, sched,    
                                      epochs, device)

my_train_help.save_checkpointpth(model,path,strct, hidden_layer, dropout ,lr)

print("Model Trained")