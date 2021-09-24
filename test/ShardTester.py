'''
Created on Sep 14, 2021

@author: thomas
'''
#import src for tests
import sys
from imageNetMapper import ImageNetMapper

sys.path.append('../src')

import unittest
import imageNetMapper
import os
import numpy.random as random
import tempfile
import imageNetProvider 
import importlib
from torch.utils.data import DataLoader
import filecmp
 
class ShardTester(unittest.TestCase):

    def setUp(self):
        self.tempFolder = tempfile.TemporaryDirectory()
        importlib.reload(imageNetProvider)
        importlib.reload(imageNetMapper)
    
    def tearDown(self):
        self.tempFolder.cleanup()
    
    def test_ShardCreation(self):
        # This only tests, whether shards can be written (and checks, whether files were created.
        imageFolder = os.path.join('Data','Images');
        files = os.listdir(imageFolder)        
        classes = random.permutation(len(files))
        mapping = {os.path.join(imageFolder,x) : y for (x,y) in zip(files,classes)}      
        imageNetMapper.buildShardsFromFolder(os.path.join('Data','Images'), mapping, self.tempFolder.name, 'Shards', maxcount=3)        
        filesInTempFolder = os.listdir(self.tempFolder.name)
        assert len(filesInTempFolder) == 4
        for file in filesInTempFolder:
            assert file.startswith('Shards')            
    
    
    def test_ShardReading(self):
        pictureNames = {'Data/Images/' + x for x in {'PiC1','Pic10','PiC11','Pic2','Pic3','Pic4','Pic5','Pic6','Pic7','Pic8','Pic9'}}
        shardNames = os.path.join('Data','Shards',"Shards{0..3}.tar")
        prov = imageNetProvider.imageNetProvider(shardNames, 2);
        loader = DataLoader(prov)
        #The first batch should be of size 2. since we have multiple workers,
        # on the data input the batch size is at most
        #batch = next(loader)
        #assert len(batch[0])== 2
        #lets see if all keys have been loaded
         
        for batch in loader:
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch['__key__']:
                assert key in pictureNames
                pictureNames.remove(key)            
                        
        assert len(pictureNames) == 0            
   
    def test_ShardReadingMultipleProcesses(self):
        pictureNames = {'Data/Images/' + x for x in {'PiC1','Pic10','PiC11','Pic2','Pic3','Pic4','Pic5','Pic6','Pic7','Pic8','Pic9'}}
        shardNames1 = os.path.join('Data','Shards',"Shards{0..1}.tar")
        shardNames2 = os.path.join('Data','Shards',"Shards{2..3}.tar")
        prov = imageNetProvider.imageNetProvider([shardNames1,shardNames2], 2);
        loader = DataLoader(prov)
        #The first batch should be of size 2. since we have multiple workers,
        # on the data input the batch size is at most
        #batch = next(loader)
        #assert len(batch[0])== 2
        #lets see if all keys have been loaded
         
        for batch in loader:
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch['__key__']:
                assert key in pictureNames
                pictureNames.remove(key)            
                        
        assert len(pictureNames) == 0   
    
    def test_ShardProcessing(self):
        imageFolder = os.path.join('Data','Images');
        files = os.listdir(imageFolder)        
        classes = random.permutation(len(files))
        mapping = {os.path.join(imageFolder,x) : y for (x,y) in zip(files,classes)}
        
        imageNetMapper.buildShardsFromFolder(os.path.join('Data','Images'), mapping, self.tempFolder.name, 'Shards', maxcount=3)
        
        filesInTempFolder = os.listdir(self.tempFolder.name)
        assert len(filesInTempFolder) == 4
        for file in filesInTempFolder:
            assert file.startswith('Shards')            
        # Thats writing the shards done. Now read them
        fileBases = {os.path.join(imageFolder,os.path.splitext(file)[0]) for file in files}
        shardNames = os.path.join(self.tempFolder.name,"Shards{0..3}.tar")
        prov = imageNetProvider.imageNetProvider(shardNames, 2);
        loader = DataLoader(prov,batch_size=2)

       #batch = prov.getBatch();
        #The first batch should be of size 2. since we have multiple workers,
        # on the data input the batch size is at most
        #assert len(batch[0])== 2
        #lets see if all keys have been loaded
         
        for batch in loader: 
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch["__key__"]:
                assert key in fileBases
                fileBases.remove(key)            
                        
        assert len(fileBases) == 0
            
    def test_DatasetMapping(self):
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = ImageNetMapper()
        mapper.extractAndPackTrainData(os.path.join('Data','Bundle.tar'),os.path.join('Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        pass   

    def test_Mapping(self):
        mapper = ImageNetMapper()
        mapper.createInstanceToClassFromSynsetInfo(os.path.join('Data','meta.mat'));
        assert(mapper.idmap['Part1'] == '1')
        assert(mapper.idmap['Part4'] == '4')
         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testShardProcessing']
    unittest.main()
    #cleanup        
