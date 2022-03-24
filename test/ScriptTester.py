from dataset_sharding import parse_args
from dataset_sharding import main as shard
import unittest
import tempfile 

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
        shard(commandlineArgs.split())