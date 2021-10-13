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

def buildShardsForTrainingDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False):
        mapper = ImageNetMapper();
        if inMemory:
            mapper.extractAndPackTrainDataInMemory(trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, preprocess)
        else:
            mapper.extractAndPackTrainData(trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, preprocess)
            


def benchmarkIOSpeeds(DataSet): 
    benchmarkReader(DataSet, IOTesters.copyAndLoad)
    benchmarkReader(DataSet, IOTesters.pureWDSRead)
    benchmarkReader(DataSet, IOTesters.wdsWithWorkers)
    benchmarkReader(DataSet, IOTesters.wdsWithWorkersAndBatches)

    