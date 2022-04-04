'''
Created on Sep 20, 2021

@author: thomas
'''

import webdataset as wds
import os.path as path
import time
from torch.utils.data import DataLoader
import tempfile
import shutil
import re
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

ShardPattern = re.compile('(.*)\{([0-9]+)\.\.([0-9]+)\}(.*)')

image_transformations = transforms.Compose([transforms.ToTensor()])  


def benchmarkReader(datasetFile, readingFunction, preprocess = None):
    '''
    This function benchmarks the IO speed on a given dataset using the provided reading function.
    The reading function is assumed to read in the whole dataset, without further processing the data
    and only takes a single argument, the provided dataset file location
    
    Parameters:
    datasetFile:        A single file containing the entire dataset
    readingfunction:    A function of the form readData(dataSetFile), that needs exactly one input argument  
                        The data of the dazaset can (and should) be directly discarded after reading. 
    '''
    # First we get the size of the dataset
    res = ShardPattern.match(datasetFile)
    if not res == None:
        start = int(res.groups()[1])
        end = int(res.groups()[2])
        padding = str(len(res.groups()[1]))
        dsSize = 0;
        for i in range(start,end+1):
            numberstring = ("{:0" + padding + "d}").format(i)
            dsFile = re.sub(ShardPattern,r'\g<1>' + numberstring + r'\g<4>',datasetFile)
            dsSize+= path.getsize(dsFile)
        pass
    else:
        dsSize = path.getsize(datasetFile)
    starttime = time.perf_counter()
    itemsTouched = readingFunction(datasetFile,preprocess)
    totaltime = time.perf_counter() - starttime;
    datarate = dsSize / totaltime / 1e6
    print("The IO speed for " + readingFunction.__name__ + " was {:0.4f} Mb/s".format(datarate) )
    print("{:} items were read".format(itemsTouched) )
    print("The IO speed per Item is " + readingFunction.__name__ + " was {:0.8f} s/Item".format(totaltime/itemsTouched))

def checkKeys(keysA, keysB):
    if len(keysB) < len(keysA):
        for key in keysB:
            if key in keysA:
                return True 
    else:
        for key in keysA:
            if key in keysB:
                return True
    return False

def copyAndLoad(DataSetFile,preprocess =None):
    '''
    First copy the Dataset and then read it from local disc
    '''    
    # Copy over
    itemsTouched = 0
    tempDir = tempfile.gettempdir()
    tempFile = path.join(tempDir,'DSFile.tar')
    shutil.copy(DataSetFile,tempFile)
    # and load    
    dataset = wds.WebDataset(tempFile)
    for element in dataset:        
        if len(element) > 0:
            itemsTouched+=1
    return itemsTouched    

def pureWDSRead(DatasetFile, preprocess =None):
    '''
    Read in the Dataset purely with WDS
    '''    
    itemsTouched = 0
    dataset = wds.WebDataset(DatasetFile)
    for element in dataset:        
        if len(element) > 0:
            itemsTouched+=1
              
    return itemsTouched

def wdsWithWorkers(DatasetFile, preprocess =None):
    '''
    Read in the Dataset with WDS using multiple workers from pyTorch
    '''   
    itemsTouched = 0     
    dataset = wds.WebDataset(DatasetFile)
    for element in dataset:        
        if len(element) > 0:
            itemsTouched+=1
    return itemsTouched


def pyTorchImageNet(DataSetFolder,preprocess =None):
    dataset = ImageNet(DataSetFolder)
    dataloader = DataLoader(dataset) 
    itemsTouched = 0   
    for element in dataset:        
        if len(element) > 0:
            itemsTouched+=1
              
    return itemsTouched

def testWDSDecode(DataSetFile, preprocess):
    if(preprocess == None):
        dataset = wds.WebDataset(DataSetFile).decode('pil').to_tuple('jpg','cls')
    else:
        dataset = wds.WebDataset(DataSetFile).to_tuple('jpg','cls').map_tuple(preprocess,lambda x:x)
    itemsTouched = 0   
    for element in dataset:        
        if len(element) > 0:
            itemsTouched+=1        
    return itemsTouched
    
def testWDSDecodeWithDL(DataSetFile, preprocess):
    if(preprocess == None):
        dataset = wds.WebDataset(DataSetFile).decode('pil').to_tuple('jpg','cls').map_tuple(image_transformations, lambda x:x)
    else:
        dataset = wds.WebDataset(DataSetFile).to_tuple('jpg','cls').map_tuple(preprocess,lambda x:x).map_tuple(image_transformations, lambda x:x)
    itemsTouched = 0
    dl = DataLoader(dataset, num_workers=4, batch_size=8)
       
    for element,cls in dl:
        for item in element:        
            itemsTouched+=1        
    return itemsTouched
    
    
    
    