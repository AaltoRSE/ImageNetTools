'''
Created on Sep 28, 2021

@author: Thomas Pfau
'''

from imageNetMapper import ImageNetMapper
import IOTesters

def buildShardsForTrainingDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False):
        mapper = ImageNetMapper();
        if inMemory:
            mapper.extractAndPackTrainDataInMemory(trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, preprocess)
        else:
            mapper.extractAndPackTrainData(trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, preprocess)
            


def benchmarkIOSpeeds(DataSet): 
    IOTesters.benchMarkReader(DataSet, IOTesters.pureWDSRead)
    IOTesters.benchMarkReader(DataSet, IOTesters.copyAndLoad)
    IOTesters.benchMarkReader(DataSet, IOTesters.wdsWithWorkers)
    IOTesters.benchMarkReader(DataSet, IOTesters.wdsWithWorkersAndBatches)

    