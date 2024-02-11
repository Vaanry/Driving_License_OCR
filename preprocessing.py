import os
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
import torch
from torchvision import transforms
import whatimage
import wand.image

def folder_scanner(folder):
    """Recursively iterates through all subfolders and collects a list of files with full paths.

    Parameters
    ----------
    folder : str
        The root folder.

    Returns
    -------
    list
        A list of whole files in the folder with full paths.
    """
    
    result = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            result.extend(folder_scanner(path))
        else:
            result.append(path)
    return result


def pdftoimg(path, poppler_path, folder=None):
    """Converts a PDF to a JPG image and saves it. Skips multi-page files.

    Parameters
    ----------
     path : str
        The path to the PDF file.
        
    poppler_path : str
        The path to the Poppler utility.
        
    folder : str, optional
        The target folder for saving the image. If provided, checks for the existence of an already converted file.
    Returns
    -------
    None

    Notes
    -----
    If the PDF is a multi-page file, this file will be skipped.
    If the PDF file is already exist in folder, no action is taken.
    """
    if path.split('.')[-1] == 'pdf':
        newfile = path.replace('.pdf', '.jpg')
        if folder:
            if newfile in os.listdir(folder):
                return
        try:
            im1 = convert_from_path(path, poppler_path=poppler_path)[0]
            im1.save(newfile)
        except PDFPageCountError:
            return
        
    
def img_files(total_files_list):
    """
    Takes a list of files as input and returns a list of image files only. Converts HEIC to JPG.

    Parameters
    ----------
    total_files_list : list of str
        A list of file paths.

    Returns
    -------
    list of str
        A list of image file paths.
    """
    img_files_list = []
    
    for file in total_files_list:
        with open(file, 'rb') as f:
            data = f.read()
            
        fmt = whatimage.identify_image(data)
        
        if fmt:
            if fmt != 'heic':
                img_files_list.append(file)
            elif fmt == 'heic':
                newfile = file.replace(".HEIC", ".jpg")
                if newfile not in total_files_list:
                    with wand.image.Image(filename=file) as img:        
                        img.format='jpg'
                        img.save(filename=newfile)
                    img_files_list.append(newfile)  
                    
    return img_files_list
    

def get_type(model, img_path):
    """
    Classifies whether the given image is a Driver License ID or another document.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ResNet model.
        
    img_path : str
        The path to the image file.

    Returns
    -------
    str
        Image type prediction: 'id' for Driver License ID or 'other' for another document.
    """
    class_names = ['id', 'other']
    
    # Save the current training mode and switch to evaluation mode
    was_training = model.training
    model.eval()   
    
    # Determine the device to use (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define data transformations for the image
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load and transform the image
    img = Image.open(img_path).convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    # Make a prediction using the model
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    # Restore the original training mode
    model.train(mode=was_training)
    
    return class_names[preds[0]]


def get_position(model, img_path):
    """
    Defines the document position.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ResNet model.
        
    img_path : str
        The path to the image file.

    Returns
    -------
    str
        Image type position: 'anticlockwise', 'clockwise', 'right', or 'upsidedown'.
    """
    class_names = ['anticlockwise', 'clockwise', 'right', 'upsidedown']
    
    # Save the current training mode and switch to evaluation mode
    was_training = model.training
    model.eval()   
    
    # Determine the device to use (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define data transformations for the image
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load and transform the image
    img = Image.open(img_path).convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    # Make a prediction using the model
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    # Restore the original training mode
    model.train(mode=was_training)
    
    return class_names[preds[0]]


def rotation(model, img_path):
    """
    Flips the image upright.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ResNet model.
        
    img_path : str
        The path to the image file.

    Returns
    -------
    PIL.Image.Image
        Image rotated to the correct position.
    """
    from .preprocessing import get_position
    
    position = get_position(model, img_path)
    angle = {'right': 0,
             'clockwise': 90,
             'upsidedown': 180,
             'anticlockwise': 270}
    
    # Open the image and rotate it based on the determined position
    rotated_img = Image.open(img_path).rotate(angle[position], expand=True)
    
    return rotated_img


def get_state(model, img_path):
    """
    Classifies the state of an ID.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ResNet model.
        
    img_path : str
        The path to the image file.

    Returns
    -------
    str
        State prediction based on the given ID image.
    """
    class_names = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Columbia', 'Connecticut',
                   'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
                   'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                   'Mississippi', 'Missouri', 'Montana', 'New York', 'Nebraska', 'Nevada', 'New Hampshire',
                   'New Jersey', 'New Mexico', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
                   'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
                   'Vermont', 'Virgin Islands', 'Virginia', 'Washington', 'Wyoming', 'West Virginia', 'Wisconsin']
    
    # Save the current training mode and switch to evaluation mode
    was_training = model.training
    model.eval()    
    
    # Define data transformations for the image
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Determine the device to use (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load and transform the image
    img = Image.open(img_path).convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    # Make a prediction using the model
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    # Restore the original training mode
    model.train(mode=was_training)
    
    return class_names[preds[0]]