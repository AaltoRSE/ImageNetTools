'''
Created on Apr 5, 2022

@author: thomas
'''

from .ImageNetTools import IOTesters 
from preprocess import ByteToPil
from .ImageNetTools import imageNetTransformations  
import sys

preprocess = imageNetTransformations.image_transformations


unprocImageNetID = 'ImageNetTrain{0..251}.tar'
procDSLocal = '/tmp/Proc/' + unprocImageNetID;
procDSNetwork = sys.argv[1] + unprocImageNetID; 
procImageNetID = 'ImageNetTrain{0..146}.tar'
unprocDSLocal = '/tmp/Unproc/ImageNetTrain{0..146}.tar';
unprocDSNetwork = sys.argv[2] + procImageNetID; 


BasicIMLocal = '/tmp/imagenet'

def runWDSLoadProcessed(procDSLocal,procDSNetwork):
    print('Testing local wds read with 4 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False)
    print('Testing local wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)
    print('Testing network wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)

def runDefaultImageNet(DataSetFolder):
    IOTesters.benchmarkReader(procDSNetwork, IOTesters.pyTorchImageNet, preprocess=preprocess)
    IOTesters.benchmarkReader(procDSNetwork, IOTesters.pyTorchImageNet, preprocess=preprocess, num_workers=12)

def runWDSLoadUnprocessed(unprocDSLocal,unprocDSNetwork):
    print('Testing local wds read with 4 workers and preprocessed data:')    
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True)
    print('Testing local wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)
    print('Testing network wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(unprocDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)


def runBenchmark():
    runWDSLoadUnprocessed(procDSLocal, procDSNetwork)
    runDefaultImageNet(BasicIMLocal)
    runWDSLoadProcessed()


runBenchmark()