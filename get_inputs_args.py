


def get_inputs_args():
    import argparse
    from torchvision import datasets, transforms, models
    parser = argparse.ArgumentParser(description = 'This hold all the information necessary to parse the command line into python data type, the inputs includes the directory to the file folder that contains the images, the file itself, and model names')
    parser.add_argument('--dir', type=str, default='ImageClassifier/flowers/', help='path to folder images')
    parser.add_argument('--arch', type=str, default=models.densenet121(pretrained=True), choices=[models.vgg19(pretrained=True), models.densenet121(pretrained=True)], help='CNN architecture to use')
    parser.add_argument('--flowerfile', type=str, default='cat_to_name.json', help='text file of the name of flower')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learnrate', type=float, default=0.003, help='learning rate for the weight update of the network')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Network computing power device location')
    parser.add_argument('--save_dir', type=str, default = 'checkpoint.pth', help='Directory where the model is saved')
    parser.add_argument('--image_dir', type=str, default='ImageClassifier/flowers/test/12/image_03994.jpg', help='Single testing Image directory')
    parser.add_argument('--topk', type=int, default=3, help='Top classes we are interested in')
    
    return parser.parse_args()
    
    
    
def check_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg, 
    assumes you defined all three command line arguments as outlined in 
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     dir =", in_arg.dir, 
              "\n    arch =", in_arg.arch, "\n flowerfile =", in_arg.flowerfile, "\n hidden =", in_arg.hidden, "\n epochs =", in_arg.epochs, "\n learnrate =", in_arg.learnrate, 
              "\n device =", in_arg.device, "\n save_directory =", in_arg.save_dir,  "\n Testing Image directory =", in_arg.image_dir, "\n TopK =", in_arg.topk
             
             
             )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    