'''
Created on Sep 28, 2021

@author: Thomas Pfau
'''

from .imageNetMapper import ImageNetMapper
from .IOTesters import benchmarkReader
from .IOTesters import pureWDSRead 
from .IOTesters import copyAndLoad 
from .IOTesters import wdsWithWorkers 
from .IOTesters import wdsWithWorkersAndBatches  
from .imageNetTransformations import image_transformations as default_transformations

#def buildShardsForTrainingDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False, filePattern=None, groundTruthBaseName=None):
#        mapper = ImageNetMapper();
#        if inMemory:
#            mapper.extractAndPackTrainDataInMemory(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName)
#        else:
#            mapper.extractAndPackTrainData(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName)
            

def buildShardsForDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False, filePattern=None, groundTruthBaseName=None):
        mapper = ImageNetMapper();
        if inMemory:
            mapper.extractAndPackDataInMemory(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName)
        else:
            mapper.extractAndPackData(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName)



def benchmarkIOSpeeds(DataSet): 
    benchmarkReader(DataSet, IOTesters.copyAndLoad)
    benchmarkReader(DataSet, IOTesters.pureWDSRead)
    benchmarkReader(DataSet, IOTesters.wdsWithWorkers)
    benchmarkReader(DataSet, IOTesters.wdsWithWorkersAndBatches)

    