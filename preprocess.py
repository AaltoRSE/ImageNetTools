'''
Created on Mar 31, 2022

@author: thomas
'''

from io import BytesIO 
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

image_transformations = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize,
                                 transforms.ToPILImage()
                                 ])   


def preprocess(binary_data):
    # load the binary data into an image format.
    f = BytesIO(binary_data)
    # Convert it into an image, so that we can apply the transformations
    image = Image.open(f)
    #Convert to RGB
    image = image.convert('RGB')
    # Apply the transformations, 
    transformed = image_transformations(image);
    # create a BytesIO Buffer to write the tensor to.
    data = BytesIO()
    # save compressed image into buffer. This is the best we get with a lossless compression.
    #transformed.save(data,'PNG')
    #The following line would generate byte-code to the file. 
    data.write(transformed.tobytes())   
    # return the contents of the buffer, which is the binary object.
    # when now using the data, you will need to interpret it as a tensorflow
    # data = BytesIO(element) 
    # restored_tensor = torch.load(data)    
    return data.getvalue()


def ByteToPil(img):
    data = BytesIO(img).read()
    im = Image.frombytes('RGB',(256,256),data)
    return im

