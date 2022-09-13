# Imports here
#matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import numpy as np

# getting images directory


# Building and training the network


def myModel(chosen_model, hidden, epochs, learnrate, device):
    model = chosen_model
    for param in model.parameters():
        param.requires_grad = False
        
        input = 1024
        hidden = 256
        output_size = 102
        pr_dropout = 0.2
        epochs = 5
        learnrate = 0.003
        
        model.to(device)
        
        modelSequence = nn.Sequential(OrderedDict([
            
            ('fc1', nn.Linear(input, hidden)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(pr_dropout)),
            ('fc2', nn.Linear(hidden, output_size)),
            ('output', nn.LogSoftmax(dim=1))
            
        ]))
        
    
    model.classifier = modelSequence
    
    model.to(device)
    return model



def learning(model, train_loader, epochs, print_every, device='cuda'):
    
    criterion = nn.NLLLoss().to(device)
    optimizer =  optim.Adam(model.classifier.parameters(), lr=0.003) 
    steps =0
    running_loss = 0
    model.to(device)
    
    for epoch in range(epochs):
        
       
        
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            # Flattening the images
            
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                print("Epoch: {}/{}".format(epoch+1, epochs),
                       "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
                model.train()


  

    





def testaccuracy(testloader, model):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Accuracy: {accuracy / len(testloader):.3f}")
    model.train()            
    
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,  map_location = ('cuda' if(device =='cuda') else 'cpu'))
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_index']
    
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #Resizing the image
    image_rate = image.size[1] / image.size[0]
    image = image.resize((256, int(image_rate*256)))
    height = image.size[1]
    width = image.size[0]
    
    #Cropping the image
    height = image.size[1]*0.5
    width = image.size[0]*0.5
    image = image.crop((width - 112,
                        height - 112,
                        width + 112,
                        height + 112))
    
    
    image = np.array(image)
    image = image / 255
    
    #Normalizing the image
    average = np.array([0.485, 0.456, 0.406])
    standard_deviation = np.array([0.229, 0.224, 0.225])
    
    image = (image - average) / standard_deviation
    
    #Re-ordering color-channel to be the first dimension to feed into Pytorch tensor
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image



def centerCoords(image, crop_size = (244, 244)):
    xwidth, xheight = image.size
    ywidth, yheight = crop_size
    left = (xwidth - ywidth) / 2
    top = (xheight - yheight) / 2
    right = (xwidth + ywidth) / 2
    bottom = (xheight + yheight) / 2
    
    return (left, top, right, bottom)
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
        
        image = image.to(device)

    logps = model.forward(image.unsqueeze(0).float())
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    return [y.item() for y in top_p.data[0]], [x.item() for x in top_class.data[0]]


def result_visual(image_path, model):
    pred_class = [cat_to_name[str(x)]for x in classes]
    fig, (ax1, ax2) = plt.subplots(figsize=(5, 6), nrows=2)
    with Image.open(image_path) as visual:
        ax1.imshow(visual)
    ax1.axis('off')
    ax1.set_title(cat_to_name[image_path.split('/')[2]])
    
    y_ax = np.arange(len(pred_class))
    ax2.set_yticks(y_ax)
    ax2.set_yticklabels(pred_class)
    ax2.barh(y_ax, class_probs)
    ax2.invert_yaxis()
    