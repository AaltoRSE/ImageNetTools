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

ShardPattern = re.compile('(.*)\{([0-9]+)\.\.([0-9]+)\}(.*)')

def benchmarkReader(datasetFile, readingFunction):
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
    itemsTouched = readingFunction(datasetFile)
    totaltime = time.perf_counter() - starttime;
    datarate = dsSize / totaltime / 1e6
    print("The IO speed for " + readingFunction.__name__ + " was {:0.4f} Mb/s".format(datarate) )
    print("{:} items were read".format(itemsTouched) )

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

def copyAndLoad(DataSetFile, checkedkey = ['jpeg','tar']):
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
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset)    
    for element in dataloader:        
        if checkKeys(checkedkey,element.keys()):
            itemsTouched+=1
        
    return itemsTouched    

def pureWDSRead(DatasetFile, checkedkey = ['jpeg','tar']):
    '''
    Read in the Dataset purely with WDS
    '''    
    itemsTouched = 0
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset)    
    for element in dataloader:        
        if checkKeys(checkedkey,element.keys()):
            itemsTouched+=1  
    return itemsTouched

def wdsWithWorkers(DatasetFile, checkedkey = ['jpeg','tar']):
    '''
    Read in the Dataset with WDS using multiple workers from pyTorch
    '''   
    itemsTouched = 0     
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset,num_workers = 4)    
    for element in dataloader:        
        if checkKeys(checkedkey,element.keys()):
            itemsTouched+=1
    return itemsTouched

def wdsWithWorkersAndBatches(DatasetFile, checkedkey = ['jpeg','tar']):    
    '''
    Read in the Dataset with WDS using multiple workers and a larger batch_size from pyTorch
    '''
    itemsTouched = 0    
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset,num_workers = 4,batch_size=100)
    try:    
        for element in dataloader:        
            if checkKeys(checkedkey,element.keys()):
                itemsTouched+=1
    except Exception as e:
        #This will happen with the last element of some sets, or if there are further items which are not jpegs in it...
        print(e)
    return itemsTouched

def pyTorchImageNet(DataSetFolder,checkedkey = ['jpeg','tar']):
    dataset = ImageNet(DataSetFolder)
    dataloader = DataLoader(dataset) 
    itemsTouched = 0   
    for element in dataloader:        
        if checkKeys(checkedkey,element.keys()):
            itemsTouched+=1
              
    return itemsTouched
