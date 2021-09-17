'''
Created on Sep 14, 2021

@author: thomas
'''
#import src for tests
import sys

sys.path.append('../src')

import unittest
import imageNetMapper
import os
import numpy.random as random
import tempfile
import imageNetProvider 
import time
import importlib

tempFolder = tempfile.TemporaryDirectory()
 
class ShardTester(unittest.TestCase):

    def setUp(self):
        importlib.reload(imageNetProvider)
        importlib.reload(imageNetMapper)

    def test_ShardProcessing(self):
        imageFolder = os.path.join('Data','Images');
        files = os.listdir(imageFolder)        
        classes = random.permutation(len(files))
        mapping = {os.path.join(imageFolder,x) : y for (x,y) in zip(files,classes)}
        
        imageNetMapper.buildShardsFromFolder(os.path.join('Data','Images'), mapping, tempFolder.name, 'Shards', maxShardFiles=3)
        
        filesInTempFolder = os.listdir(tempFolder.name)
        assert len(filesInTempFolder) == 4
        for file in filesInTempFolder:
            assert file.startswith('Shards')            
        # Thats writing the shards done. Now read them
        fileBases = {os.path.join(imageFolder,os.path.splitext(file)[0]) for file in files}
        shardNames = os.path.join(tempFolder.name,"Shards{0..3}.tar")
        prov = imageNetProvider.imageNetProvider(shardNames, 2);
        batch = prov.getBatch();
        #The first batch should be of size 2. since we have multiple workers,
        # on the data input the batch size is at most
        assert len(batch[0])== 2
        #lets see if all keys have been loaded
         
        while not batch == None:
            assert len(batch[0])>= 1 # we can't make a stronger assertion here.
            for key in batch[0]:
                assert key in fileBases
                fileBases.remove(key)            
            print(batch[2])
            print(batch[0])
            batch = prov.getBatch()
        
        assert len(fileBases) == 0
            
        
        

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testShardProcessing']
    unittest.main()
    #cleanup        
    tempFolder.cleanup();
