'''
Created on Sep 20, 2021

@author: thomas
'''

import webdataset as wds
import os.path as path
import time
from torch.utils.data import DataLoader
import imageNetProvider
import tempfile
import shutil

def benchMarkReader(datasetFile, readingFunction):
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
    dsSize = path.getsize(datasetFile)
    starttime = time.perf_counter()
    readingFunction(datasetFile)
    totaltime = time.perf_counter() - starttime;
    datarate = dsSize / totaltime / 1e6
    print("The IO speed was {:0.4f} Mb/s".format(datarate) )

def copyAndLoad(DataSetFile, checkedkey = 'jpeg'):
    '''
    First copy the Dataset and then read it from local disc
    '''    
    # Copy over
    tempDir = tempfile.gettempdir()
    tempFile = path.join(tempDir,'DSFile.tar')
    shutil.copy(DataSetFile,tempFile)
    # and load    
    dataset = wds.WebDataset(tempFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset)    
    for element in dataloader:        
        try:
            #We want to make sure, that the jpeg data is actually loaded
            temp = element[checkedkey]
        except Exception as e:
            print(e)    

def pureWDSRead(DatasetFile, checkedkey = 'jpeg'):
    '''
    Read in the Dataset purely with WDS
    '''    
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset)    
    for element in dataloader:        
        try:
            #We want to make sure, that the jpeg data is actually loaded
            temp = element[checkedkey]
        except Exception as e:
            print(e)        
        
def wdsWithWorkers(DatasetFile, checkedkey = 'jpeg'):
    '''
    Read in the Dataset with WDS using multiple workers from pyTorch
    '''    
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset,num_workers = 4)    
    for element in dataloader:        
        try:
            #We want to make sure, that the jpeg data is actually loaded
            temp = element[checkedkey]
        except Exception as e:
            print(e)

def wdsWithWorkersAndBatches(DatasetFile, checkedkey = 'jpeg'):    
    '''
    Read in the Dataset with WDS using multiple workers and a larger batch_size from pyTorch
    '''
    dataset = wds.WebDataset(DatasetFile)
    #Build the wrapper since this will be closer to what we will have
    dataloader = DataLoader(dataset,num_workers = 4,batch_size=100)
    try:    
        for element in dataloader:        
            try:
                #We want to make sure, that the jpeg data is actually loaded
                temp = element[checkedkey]
            except Exception as e:
                print(e)
    except Exception as e:
        #This will happen with the last element of some sets, or if there are further items which are not jpegs in it...
        print(e)

def readWithProcess(DatasetFile, checkedkey = 'jpeg'):
    '''
    Read in the Dataset using wds from an external processes.
    If DataSetFile consists of multiple filenames, one process per filename will be started to parallelise IO.
    '''        
    prov = imageNetProvider.imageNetProvider(DatasetFile, batchSize=100)
    dataloader = DataLoader(prov,batch_size=100)

    for element in dataloader:
        try:
            #We want to make sure, that the jpeg data is actually loaded
            temp = element[checkedkey]
        except Exception as e:
            print(e)   
        
def readWithMultipleProcesses(DatasetFile, checkedkey = 'jpeg'):
    '''

    '''            
    prov = imageNetProvider.imageNetProvider(DatasetFile, batchSize=100)
    dataloader = DataLoader(prov,batch_size=100)

    for element in dataloader:
        try:
            #We want to make sure, that the jpeg data is actually loaded
            temp = element[checkedkey]
        except Exception as e:
            print(e)   
