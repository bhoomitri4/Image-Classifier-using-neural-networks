import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import my_train_help

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='ImageClassifier/flowers/test/102/image_08004.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint
hidden_layer = pa.hidden_units
dropout = pa.dropout

#dataloaders, dataset_sizes , image_datasets = my_train_help.data_loading(path_image)

my_train_help.load_model(path,hidden_layer , dropout)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
top_probs , labels ,top_flowers= my_train_help.predict( input_img, path, hidden_layer, dropout, number_of_outputs)  

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(top_flowers[i], top_probs[i]))
    i += 1
     