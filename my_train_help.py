#importing libraries
import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn
import torch.optim as optm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

arch = {"vgg19":25088,
        "densenet121":1024,
        "alexnet":9216}
model = ''


def data_loading(loc = "ImageClassifier/flowers"):
    data_dir = loc
    #train_dir = data_dir + '/train'
    #valid_dir = data_dir + '/valid'
    #test_dir = data_dir + '/test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dirs = {'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir}
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders , dataset_sizes,image_datasets

#label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
    
def model_setup(strct = 'vgg19' , dropout = 0.5 , hidden_layer = 4096,lr = 0.001,device = 'cuda'):
    if strct == 'vgg19':
        model = models.vgg19(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout)),
                          ('func1', nn.Linear(25088, hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('func2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif strct == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout)),
                          ('func1', nn.Linear(1024, hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('func2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif strct == 'alexnet':
        model = models.alexnet(pretrained = True)
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout)),
                          ('func1', nn.Linear(9216, hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('func2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    for params in model.parameters():
        params.requires_grad = False
    model.classifier = classifier
   
    criteria = nn.NLLLoss()
    optimizer = optm.Adam(model.classifier.parameters(), lr)
    sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    eps=10
    if torch.cuda.is_available() and device == 'cuda':
            model.cuda()
    return model,criteria , optimizer , sched  



def train_model(model, criteria, optimizer, scheduler,    
                                      num_epochs=25, device='cuda'):
    dataloaders , dataset_sizes ,image_datasets= data_loading('ImageClassifier/flowers')
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpointpth(model,path = 'checkpoint.pth',strct = 'vgg19', hidden_layer = 4096 , dropout= 0.5 ,lr=0.001 ):
    dataloaders , dataset_sizes ,image_datasets= data_loading('ImageClassifier/flowers')
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'arch': strct,
            'hidden_layer' : hidden_layer,
            'dropout' : dropout,
            'lr' : lr,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx},
              path)
    
def load_model(checkpoint_path,hidden_layer , dropout):
    chpt = torch.load(checkpoint_path)
    
    strct = chpt['arch']
    hidden_layer1 = chpt['hidden_layer']
    dropout = chpt['dropout']
    lr=chpt['lr']  
    
    model ,_,_,_ =  model_setup(strct  ,dropout  , hidden_layer ,lr ,device = 'cuda')    
    
    model.class_to_idx = chpt['class_to_idx']
    
    
    # Create the classifier
    #classifier = nn.Sequential(OrderedDict([
                          #('fc1', nn.Linear(25088, 4096)),
                          #('relu', nn.ReLU()),
                          #('fc2', nn.Linear(4096, 102)),
                          #('output', nn.LogSoftmax(dim=1))
                          #]))
    # Put the classifier on the pretrained network
    #model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    
    
    
def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img



def predict(image_path,  checkpoint_path , hidden_layer , dropout,top_num=5):
    chpt = torch.load(checkpoint_path)
    
    strct = chpt['arch']
    hidden_layer1 = chpt['hidden_layer']
    dropout = chpt['dropout']
    lr=chpt['lr']  
    
    model ,_,_,_ =  model_setup(strct  ,dropout  , hidden_layer ,lr ,device = 'cuda')
    model.class_to_idx = chpt['class_to_idx']
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels,top_flowers
    # TODO: Implement the code to predict the class from an image file
    
    





    
    