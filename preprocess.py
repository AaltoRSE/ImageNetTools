'''
Created on Mar 31, 2022

@author: thomas
'''

from io import BytesIO 
from PIL import Image
import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

image_transformations = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize])   


def preprocess(binary_data):
    # load the binary data into an image format.
    f = BytesIO(binary_data)
    # Convert it into an image, so that we can apply the transformations
    image = Image.open(f)
    # Apply the transformations
    transformed = image_transformations(image);
    # create a BytesIO Buffer to write the tensor to.
    data = BytesIO()
    # save into the buffer
    torch.save(transformed,data)
    # return the contents of the buffer, which is the binary object.
    # when now using the data, you will need to interpret it as a tensorflow
    # data = BytesIO(element) 
    # restored_tensor = torch.load(data)    
    return data.get_value()