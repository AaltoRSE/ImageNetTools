'''
Created on Sep 14, 2021

@author: thomas
'''
#import src for tests


from ImageNetTools import imageNetMapper, imageNetProvider
import unittest
import os
import numpy.random as random
import tempfile
import importlib
from torch.utils.data import DataLoader
import ImageNetTools
 
class ShardTester(unittest.TestCase):

    def setUp(self):
        self.tempFolder = tempfile.TemporaryDirectory()
        importlib.reload(imageNetProvider)
        importlib.reload(imageNetMapper)
    
    def tearDown(self):
        self.tempFolder.cleanup()
    
    def test_ShardCreation(self):
        # This only tests, whether shards can be written (and checks, whether files were created.
        print('Shard creation test')
        imageFolder = os.path.join('tests','Data','Images');
        files = os.listdir(imageFolder)        
        classes = random.permutation(len(files))
        mapping = {x : y for (x,y) in zip(files,classes)}      
        imageNetMapper.buildShardsFromFolder(os.path.join('tests','Data','Images'), mapping, self.tempFolder.name, 'Shards', maxcount=3)        
        filesInTempFolder = os.listdir(self.tempFolder.name)
        assert len(filesInTempFolder) == 5
        for file in filesInTempFolder:
            assert file.startswith('Shards') or file == 'FileInfo.json'
    
    
    def test_ShardReading(self):
        print('Shard reading test')
        pictureNames = {'Data/Images/' + x for x in {'PiC1','Pic10','PiC11','Pic2','Pic3','Pic4','Pic5','Pic6','Pic7','Pic8','Pic9'}}
        shardNames = os.path.join('tests','Data','Shards',"Shards{0..3}.tar")
        prov = imageNetProvider.imageNetProvider(shardNames, 2);
        loader = DataLoader(prov)

        for batch in loader:
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch['__key__']:
                assert key in pictureNames
                pictureNames.remove(key)                                    
        assert len(pictureNames) == 0            
   
    def test_ShardReadingMultipleProcesses(self):
        print('Shard reading multi-process test')
        pictureNames = {'Data/Images/' + x for x in {'PiC1','Pic10','PiC11','Pic2','Pic3','Pic4','Pic5','Pic6','Pic7','Pic8','Pic9'}}
        shardNames1 = os.path.join('tests', 'Data','Shards',"Shards{0..1}.tar")
        shardNames2 = os.path.join('tests','Data','Shards',"Shards{2..3}.tar")
        prov = imageNetProvider.imageNetProvider([shardNames1,shardNames2], 2);
        loader = DataLoader(prov)
        for batch in loader:
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch['__key__']:
                assert key in pictureNames
                pictureNames.remove(key)                                    
        assert len(pictureNames) == 0   
    
    def test_ShardProcessing(self):
        print('Shard processing test')
        imageFolder = os.path.join('tests','Data','Images');
        files = os.listdir(imageFolder)        
        classes = random.permutation(len(files))
        mapping = {x : y for (x,y) in zip(files,classes)}
        
        imageNetMapper.buildShardsFromFolder(os.path.join('tests','Data','Images'), mapping, self.tempFolder.name, 'Shards', maxcount=3)
        
        filesInTempFolder = os.listdir(self.tempFolder.name)
        assert len(filesInTempFolder) == 5
        for file in filesInTempFolder:
            assert file.startswith('Shards') or file == 'FileInfo.json'
            
        # Thats writing the shards done. Now read them
        fileBases = {os.path.splitext(file)[0] for file in files}
        shardNames = os.path.join(self.tempFolder.name,"Shards{0..3}.tar")
        prov = imageNetProvider.imageNetProvider(shardNames, 2);
        loader = DataLoader(prov,batch_size=2)
         
        for batch in loader: 
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key in batch["__key__"]:
                assert key in fileBases
                fileBases.remove(key)            
                        
        assert len(fileBases) == 0
            
    
            
    def test_DatasetMapping(self):
        print('Dataset mapping test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackData(os.path.join('tests','Data','Bundle.tar'),os.path.join('tests','Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))
        #For now, we won't test contents (as that's tested elsewhere.

    def test_Mapping(self):
        print('Mapping test')
        mapper = imageNetMapper.ImageNetMapper()
        mapper.createInstanceToClassFromSynsetInfo(os.path.join('tests','Data','meta.mat'));
        assert(mapper.idmap['Part1'] == '1')
        assert(mapper.idmap['Part4'] == '4')
        
    def test_inMemory(self):
        print('In Memory test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackDataInMemory(os.path.join('tests','Data','Bundle.tar'),os.path.join('tests','Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))
        #For now, we won't test contents (as that's tested elsewhere.
        
    def test_TestIO(self):
        print('IO test test')
        ImageNetTools.benchmarkIOSpeeds('tests/Data/Bundle_no_tars.tar')
        
    def test_testNonTaredTrain(self):        
        print('Non Tared Train test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackData(os.path.join('tests','Data','Bundle_no_tars.tar'),os.path.join('tests','Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))
    
    def test_testNonTaredTrain_inMemory(self):        
        print('Non tared train in memory test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackDataInMemory(os.path.join('tests','Data','Bundle_no_tars.tar'),os.path.join('tests','Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))

    def test_testZippedTrain(self):      
        print('Zipped Train test')  
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackData(os.path.join('tests','Data','Bundle.tar.gz'),os.path.join('tests','Data','meta.mat'),outFolder,'TestSet',maxcount=3 )
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))

    def test_testgroundTruth(self):        
        print('Groudn truth test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackData(os.path.join('tests','Data','ValTest.tar'),os.path.join('tests','Data','groundTruth.txt'),outFolder,'TestSet',maxcount=3, groundTruthBaseName='Pic')
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))


    def test_testgroundTruthMemory(self):        
        print('Ground truth memory test')
        outFolder = os.path.join(self.tempFolder.name,'dsOutput')
        os.mkdir(outFolder)
        mapper = imageNetMapper.ImageNetMapper()
        mapper.extractAndPackDataInMemory(os.path.join('tests','Data','ValTest.tar'),os.path.join('tests','Data','groundTruth.txt'),outFolder,'TestSet',maxcount=3, groundTruthBaseName='Pic')
        assert(os.path.isfile(os.path.join(outFolder, 'TestSet2.tar')))

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testShardProcessing']
    unittest.main()
    #cleanup        
