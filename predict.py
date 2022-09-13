#from functions import *
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

in_args = get_inputs_args()
check_command_line_arguments(in_args)

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
model = load_checkpoint(in_args.save_dir, map_location)



class_probs, classes = predict(in_args.image_dir, model, map_location, in_args.topk)
print()
print("Class probability: ", max(class_probs))
print("Class name: ", classes)

#with open('ImageClassifier/cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)
    
#pred_class = [cat_to_name[str(x)] for x in classes]



