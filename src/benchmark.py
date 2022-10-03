'''
Created on Apr 5, 2022

@author: thomas
'''

from .ImageNetTools import IOTesters 
from preprocess import ByteToPil
from .ImageNetTools import imageNetTransformations  
import sys

preprocess = imageNetTransformations.image_transformations


procImageNetID = 'ImageNetTrain{0..251}.tar'
procDSLocal = '/tmp/Proc/' + procImageNetID;
procDSNetwork = sys.argv[1] + procImageNetID; 
unprocImageNetID = 'ImageNetTrain{0..146}.tar'
unprocDSLocal = '/tmp/Unproc/ImageNetTrain{0..146}.tar';
unprocDSNetwork = sys.argv[2] + unprocImageNetID; 


BasicIMLocal = '/tmp/imagenet'

def runWDSLoadProcessed(procDSLocal,procDSNetwork):
    print('Testing local wds read with 4 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False)
    print('Testing local wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)
    print('Testing network wds read with 12 workers and preprocessed data:')    
    IOTesters.benchmarkReader(procDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)

def runDefaultImageNet(DataSetFolder):
    IOTesters.benchmarkReader(DataSetFolder, IOTesters.pyTorchImageNet, preprocess=preprocess)
    IOTesters.benchmarkReader(DataSetFolder, IOTesters.pyTorchImageNet, preprocess=preprocess, num_workers=12)

def runWDSLoadUnprocessed(unprocDSLocal,unprocDSNetwork):
    print('Testing local wds read with 4 workers and unpreprocessed data:')    
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True)
    print('Testing local wds read with 12 workers and unpreprocessed data:')    
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)
    print('Testing network wds read with 12 workers and unpreprocessed data:')    
    IOTesters.benchmarkReader(unprocDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)


if sys.argv[3] == 'un':
    print('Testing unrocessed Data')
    runWDSLoadUnprocessed(unprocDSLocal, unprocDSNetwork)

if sys.argv[3] == 'def':
    print('Testig processed Data')
    runDefaultImageNet(BasicIMLocal)

if sys.argv[3] == 'proc':
    print('Tesing basic ImageNet data')
    runWDSLoadProcessed(procDSLocal, procDSNetwork)

if sys.argv[3] == '1':
    print('Testing local wds read with 4 workers and preprocessed data:')
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False)
if sys.argv[3] == '2':
    print('Testing local wds read with 12 workers and preprocessed data:')
    IOTesters.benchmarkReader(procDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)
if sys.argv[3] == '3':
    print('Testing network wds read with 12 workers and preprocessed data:')
    IOTesters.benchmarkReader(procDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=ByteToPil, dataType='img', pilData=False, num_workers=12)

if sys.argv[3] == '4':
    print("Running IOTester with pytorch imagenet dataset and defaults")
    IOTesters.benchmarkReader(BasicIMLocal, IOTesters.pyTorchImageNet, preprocess=preprocess)
if sys.argv[3] == '5':
    print("Running IOTester with pytorch imagenet dataset and 12 workers")
    IOTesters.benchmarkReader(BasicIMLocal, IOTesters.pyTorchImageNet, preprocess=preprocess, num_workers=12)

if sys.argv[3] == '6':
    print('Testing local wds read with 4 workers and unpreprocessed data:')
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True)
if sys.argv[3] == '7':
    print('Testing local wds read with 12 workers and unpreprocessed data:')
    IOTesters.benchmarkReader(unprocDSLocal, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)
if sys.argv[3] == '8':
    print('Testing network wds read with 12 workers and unpreprocessed data:')
    IOTesters.benchmarkReader(unprocDSNetwork, IOTesters.testWDSDecodeWithDL, preprocess=preprocess, dataType='jpg', pilData=True, num_workers=12)