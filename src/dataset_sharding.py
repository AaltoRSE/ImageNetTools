'''
Created on Sep 29, 2021

@author: thomas
'''

from ImageNetTools import buildShardsForFolder,buildShardsForDataset
import sys
import os
import re
import argparse
import json


def __doc__():
    '''Call by providing a config file in json format (or the individual elements as arguments to the call as -- arguments)'''

def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--conf', action='append', help= "The name of the configuration file in json format. Can contain elements representing any other argument")
    parser.add_argument('-d', '--dataSource', help='The path to the dataset file(if a tar) or folder (if its images) to shard (e.g. /path/to/folder)', type=str)
    parser.add_argument('-o', '--targetFolder', help='The output folder for the shards (e.g. /path/to/output/folder)', type=str)
    parser.add_argument('-m', '--metaDataFile', help='The path to the dataset tar file to shard (e.g. /path/to/meta/info/file)', type=str)
    parser.add_argument('-r', '--filePattern',default=None, help='The regular expression to extract the class from the files\nThe default uses the folder name before the actual file', type=str)
    parser.add_argument('-n', '--datasetName',default='Dataset', help='The Base name of the output shards', type=str)    
    parser.add_argument('-x', '--maxcount',default=100000, help='Maximium number of files per shard', type=int)
    parser.add_argument('-s', '--maxsize',default=1e9, help='Maximium size per shard', type=int)
    parser.add_argument('-y', '--inmemory',default=False, help='Whether to load the entire dataset into memory for sharding, only relevant if the original dataSource is a tar file', type=bool)
    parser.add_argument('-g', '--groundTruthBaseName',default='', help='The base name for a ground truth file to add the ID to (normally validation)\nIf not present or left empty, it wont be used. Relevant for e.g. imageNet validation data', type=str)
    parser.add_argument('-p', '--preprocess',default=None, help='A function name provided in the module preprocess that is to be used as a preprocessing function and will be applied to the binary data of each input file. This means, that the preprocessing has to take care of converting the data into a usable format and returning it into a usable binary format.', type=str)
    parser.add_argument('-e', '--seed',default=1, help='The seed used to create the permutation', type=int)
    parser.add_argument('--dataType',default='img', help='The type of data to be processed, only relevant for the key the data is stored under in the shards.', type=str)
    
    resargs = parser.parse_args(args)
    if resargs.conf is not None:
        for conf_fname in resargs.conf:
            with open(conf_fname, 'r') as f:
                parser.set_defaults(**json.load(f))
    # Reparse to overwrite config file values by values from command line.    
        resargs = parser.parse_args(args)
    if not resargs.preprocess == None:
        import preprocess
        resargs.preprocess = getattr(preprocess,resargs.preprocess)
    
    return resargs
    
def main(sysargs):
    args = parse_args(sysargs)
    os.makedirs(args.targetFolder,exist_ok=True)
    if(os.path.isdir(args.dataSource)):
        buildShardsForFolder(args.dataSource, args.metaDataFile, args.targetFolder, args.datasetName, maxcount=args.maxcount, maxsize=args.maxsize, filePattern=args.filePattern, preprocess=args.preprocess, groundTruthBaseName=args.groundTruthBaseName, dataType=args.dataType, seed=args.seed)
    else:
        buildShardsForDataset(args.dataSource, args.metaDataFile, args.targetFolder, args.datasetName, maxcount=args.maxcount, maxsize=args.maxsize, inMemory = args.inMemory, filePattern=args.filePattern, preprocess=args.preprocess, groundTruthBaseName=args.groundTruthBaseName, dataType=args.dataType, seed=args.seed)     



if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()