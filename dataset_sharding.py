'''
Created on Sep 29, 2021

@author: thomas
'''

import ImageNetTools
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
    parser.add_argument('-d', '--dataFile', help='The path to the dataset file(s) to shard (e.g. /path/to/folder)', type=str)
    parser.add_argument('-o', '--targetFolder', help='The output folder for the shards (e.g. /path/to/output/folder)', type=str)
    parser.add_argument('-m', '--metaDataFile', help='The path to the dataset tar file to shard (e.g. /path/to/meta/info/file)', type=str)
    parser.add_argument('-r', '--filePattern',default='.*?([^/]+)/[^/]*\..*', help='The regular expression to extract the class from the files\nThe default uses the folder name before the actual file', type=str)
    parser.add_argument('-n', '--datasetName',default='Dataset', help='The Base name of the output shards', type=str)    
    parser.add_argument('-x', '--maxcount',default=100000, help='Maximium number of files per shard', type=int)
    parser.add_argument('-s', '--maxsize',default=3e9, help='Maximium size per shard', type=int)
    parser.add_argument('-y', '--inmemory',default=False, help='Whether to load the entire dataset into memory for sharding', type=bool)
    parser.add_argument('-g', '--groundTruthBaseName',default='', help='The base name for a ground truth file to add the ID to (normally validation)\nIf not present or left empty, it wont be used. Should not be given for training data', type=str)

    resargs = parser.parse_args(args)
    if resargs.conf is not None:
        for conf_fname in resargs.conf:
            with open(conf_fname, 'r') as f:
                parser.set_defaults(**json.load(f))
    # Reparse to overwrite config file values by values from command line.    
        resargs = parser.parse_args(args)
    return resargs
    
def main(sysargs):
    args = parse_args(sysargs)
    os.makedirs(args.targetFolder,exist_ok=True)
    ImageNetTools.buildShardsForDataset(args.dataFile, args.metaDataFile, args.targetFolder, args.datasetName, maxcount=args.maxcount, maxsize=args.maxsize, inMemory = args.inMemory, filePattern=re.compile(args.filePattern), groundTruthBaseName=args.groundTruthBaseName)     

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()