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
 
tempFolder = tempfile.TemporaryDirectory()
 
class ShardTester(unittest.TestCase):


    def test_ShardProcessing(self):
        imageFolder = os.path.join('Data','Images');
        files = os.listdir(imageFolder)
        classes = random.permutation(len(files))
        mapping = {os.path.join(imageFolder,x) : y for (x,y) in zip(files,classes)}
        print(mapping)
        
        imageNetMapper.buildShardsFromFolder(os.path.join('Data','Images'), mapping, tempFolder.name, 'Shards', maxShardFiles=3)
        
        filesInTempFolder = os.listdir(tempFolder.name)
        assert len(filesInTempFolder) == 4
        for file in filesInTempFolder:
            assert file.startswith('Shards')            
        
    def test_

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testShardProcessing']
    unittest.main()
    #cleanup        
    tempFolder.cleanup();
