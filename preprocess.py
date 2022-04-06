'''
Created on Mar 31, 2022

@author: thomas
'''

from io import BytesIO 
from PIL import Image
import torchvision.transforms as transforms


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
    """ 
    This preprocessing function resizes the images, and produces byte-code (larger than jpg but loaded more efficiently).
    It is expected to be used with the ByteToPil function for later processing in pytorch.
    """ 
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
    """ 
    This is a helper function to convert the generated images in byte-code back to PIL images for data loading.
    It is expected to be used with webdataset datasets in a map_tuple function, where this function is applied to 
    the image data loaded:
    wds.to_tuple('img','cls').map_tuple(ByteToPil,lambda x:x).map_tuple(image_cropping, lambda x:x)
    """
    data = BytesIO(img).read()
    im = Image.frombytes('RGB',(256,256),data)
    return im

def decodeClass(cls):
    """
    By default webdataset converts str and int values into binary values. To obtain ints for processing, they need to be 
    decoded. Commonly pytorch expects an int class, so here is a simple decoder that can be used along with the BytToPil function 
    like:
    WebDataSet(FileName).to_tuple('img','cls').map_tuple(ByteToPil,decodeClass).map_tuple(image_cropping, lambda x:x).shuffle(10000)        
    """
    return int(cls) 