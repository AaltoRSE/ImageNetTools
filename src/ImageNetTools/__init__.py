'''
Created on Sep 28, 2021

@author: Thomas Pfau
'''

from .imageNetMapper import ImageNetMapper
from .IOTesters import benchmarkReader
from .IOTesters import pureWDSRead 
from .IOTesters import copyAndLoad 
from .imageNetTransformations import image_transformations as default_transformations

def buildShardsForFolder(dataFolder, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, filePattern=None, groundTruthBaseName=None, dataType='img', seed=1):        
        mapper = ImageNetMapper();
        mapper.shardDataFolder(dataFolder, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName, dataType=dataType, seed=seed)
            

def buildShardsForDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False, filePattern=None, groundTruthBaseName=None, dataType='img', seed=1):
        mapper = ImageNetMapper();
        if inMemory:
            mapper.extractAndPackDataInMemory(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName, dataType=dataType, seed=seed)
        else:
            mapper.extractAndPackData(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName, dataType=dataType, seed=seed)



def benchmarkIOSpeeds(DataSet): 
    benchmarkReader(DataSet, IOTesters.copyAndLoad)
    benchmarkReader(DataSet, IOTesters.pureWDSRead)

    