from dataset_sharding import parse_args
from dataset_sharding import main as shard
import imageNetProvider  
import unittest
import tempfile 
import os
from webdataset import WebDataset as wds
from torch.utils.data import DataLoader


class ScriptTester(unittest.TestCase):

    def setUp(self):
        self.tempFolder = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        self.tempFolder.cleanup()
        
        
    def test_shard_parser(self):
        # This only tests, whether shards can be written (and checks, whether files were created.
        commandlineArgs = "--conf testConfig -x 2"
        args = parse_args(commandlineArgs.split())
        assert args.maxcount == 2
        assert args.dataFile == "../ImageNetTools/test/Data/Bundle.tar"
    
    def test_shard_main(self):
        commandlineArgs = "--conf testConfig -x 2"
        args = parse_args(commandlineArgs.split())    
        shard(commandlineArgs.split())
        filesInTempFolder = os.listdir(args.targetFolder)
        assert len(filesInTempFolder) == 6 # we have 11 files those go into 6 nw files as a max of 2 files is permitted. 
        for file in filesInTempFolder:
            assert file.startswith(args.datasetName)
        
        #Now, test the contents.
        # Since the pictures came from Part1.tars, these will be kept in the key.
        pictureNames = {'Part1/PiC1': '1','Part4/Pic10' : '4','Part4/PiC11': '4','Part1/Pic2' : '1','Part1/Pic3': '1','Part2/Pic4': '2','Part2/Pic5' : '2','Part2/Pic6' : '2','Part3/Pic7' : '3','Part3/Pic8' : '3','Part3/Pic9' : '3'}
        shardNames = os.path.join('testOutput',"INValidation{0..5}.tar")
        ds = wds(shardNames);
        loader = DataLoader(ds)
        for batch in loader:
            assert len(batch["__key__"])>= 1 # we can't make a stronger assertion here.
            for key,cls in zip(batch['__key__'],batch['cls']):
                assert key in pictureNames
                assert cls.decode() == pictureNames[key]
                del pictureNames[key]  
                                                  
        assert len(pictureNames) == 0 
        