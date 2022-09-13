
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import numpy as np







def learn(model, trainloader, validloader, epochs, print_every, device):
    from torch import nn, optim
    criterion = nn.NLLLoss().to(device)
    optimizer =  optim.Adam(model.classifier.parameters(), lr=0.003) 
    steps =0
    running_loss = 0
    model.to(device)
    
    for epoch in range(epochs):
        
       
        
        for images, labels in trainloader:
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
                
                valid_loss=0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                
                
                
                print("Epoch: {}/{}".format(epoch+1, epochs),
                       "training loss: {:.4f}".format(running_loss/print_every), 
                      "validation loss: {:.4f}".format(valid_loss/len(validloader)),
                      "Valid_accuracy: {:.4f}".format(accuracy/len(validloader))
                     )
                running_loss = 0
                model.train()

#torch.cuda.DoubleTensor
def load_checkpoint(filepath, location):
    checkpoint = torch.load(filepath,  map_location = location)
    model=checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_index']
    for parameter in model.parameters():
        parameter.requires_grad = False
    
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


def predict(image_path, model, map_location, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
        
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')

    logps = model.forward(image.unsqueeze(0).float())
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    class_probs, classes = [y.item() for y in top_p.data[0]], [x.item() for x in top_class.data[0]]
# to be continued
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        import json
        cat_to_name = json.load(f)
    
    #pred_class = [cat_to_name[str(x)] for x in classes]
    title = cat_to_name[image_path.split('/')[3]]
    return class_probs, title


