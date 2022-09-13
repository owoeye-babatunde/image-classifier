from functions import *
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from workspace_utils import active_session                                                                                                  
from PIL import Image
import numpy as np
from get_inputs_args import *
from rets import *
import json 

# Navigating the data directory



in_args = get_inputs_args()
check_command_line_arguments(in_args)

train_dir = in_args.dir + 'train'
valid_dir = in_args.dir + 'valid'
save_dir = 'ImageClassifier/' + in_args.save_dir
test_dir = in_args.dir + 'test'

data_dir = 'ImageClassifier/flowers'
image_dir = {
                'train' : data_dir + '/train',
                'validation' : data_dir + '/valid',
                'test' : data_dir + '/test'
            }


# Transforming the images
image_transforms = {
                    'train' : transforms.Compose([
                                     transforms.RandomRotation(30),
                                     transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                      ]),

                    'test' :  transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])]),

                    'validation' : transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])
                    }

# TODO: Loading the datasets with ImageFolder
image_dataset = {
                    'train' : datasets.ImageFolder(train_dir, transform=image_transforms['train']),
                    'test' : datasets.ImageFolder(test_dir, transform=image_transforms['test']),
                    'validation' : datasets.ImageFolder(valid_dir, transform=image_transforms['validation'])
                }

# TODO: Using the image datasets and the trainforms to define the dataloaders
image_loader = {
                'train' : torch.utils.data.DataLoader(image_dataset['train'], batch_size=64, shuffle=True),
                'test' : torch.utils.data.DataLoader(image_dataset['test'], batch_size=64),
                'validation' : torch.utils.data.DataLoader(image_dataset['validation'], batch_size=64)
                }



# Opening and loading the test image


with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
    
model = myModel(in_args.arch, in_args.hidden, in_args.epochs, in_args.learnrate, in_args.device)


# Defining Hyperparameters
hyper = {
            
            'criterion' : nn.NLLLoss().to(in_args.device),
            'optimizer' : optim.Adam(model.classifier.parameters(), lr=0.003)     
        }
  

with active_session():
    learn(model, image_loader['train'], image_loader['validation'], 2, 10,  device='cuda')  
   
    
#printing validation accuracy





# Saving the trained model to checkpoint
model.class_to_idx = image_dataset['train'].class_to_idx

checkpoint = {
    'input_size': 1024,
    'output_size': 102,
    'hidden_size': in_args.hidden,
    'epoch': 2,
    'learning rate':in_args.learnrate,
    'batch_size': 64,
    'data_transforms': image_transforms['test'],
    'model':in_args.arch,
    'classifier': model.classifier,    
    'optimizer': hyper['optimizer'].state_dict,
    'criterion': hyper['criterion'],
    'model_index': model.class_to_idx,
    'state_dict': model.state_dict()
                }

torch.save(checkpoint, in_args.save_dir)

# !rm -f ~/opt/*.pth








