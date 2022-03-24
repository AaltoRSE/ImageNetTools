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
    parser.add_argument('-d', '--dataFile', help='The path to the dataset tar file to shard (e.g. /path/to/folder', type=str)
    parser.add_argument('-o', '--targetFolder', help='The output folder for the shards (e.g. /path/to/output/folder', type=str)
    parser.add_argument('-m', '--metaDataFile', help='The path to the dataset tar file to shard (e.g. /path/to/meta/info/file', type=str)
    parser.add_argument('-r', '--fileRegexp',default='.*?([^/]+)/[^/]*\..*', help='The regular expression to extract the class from the files\nThe default uses the folder name before the actual file', type=str)
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
    
# def parseOptions(configFile):
#     f = open(os.path.expandvars(configFile),'r');
#     configOptions = f.readlines();
#     f.close()
#     dsName = 'ImageNetShards'
#     maxcount=100000
#     maxsize=3e9
#     inMemory = False
#     filePattern = ImageNetTools.imageNetMapper.finalFilePattern
#     targetFolder =  os.path.expandvars('$WRKDIR/ImageNetShards');
#     trainDataFile =  os.path.expandvars('/scratch/shareddata/dldata/imagenet/imagenet21k_resized.tar.gz');
#     metaDataFile =  os.path.expandvars('/scratch/shareddata/dldata/imagenet/ILSVRC2012_devkit_t12/data/meta.mat');
#     groundTruthBaseName = False
#     for option in configOptions:
#         #Remove comments
#         if '#' in option:
#             option = option.split('#',1)[0]
#         #Skip lines without aktual assignments
#         if not '=' in option:
#             continue
#         cOption = option.split('=',1)
#         #Remove leading and trailing whitespaces.
#         optionName = cOption[0].lstrip().rstrip();
#         optionValue = cOption[1].lstrip().rstrip();
#         if optionName == 'dataFile':  
#             trainDataFile = os.path.expandvars(optionValue);
#         elif optionName == 'metaDataFile':
#             metaDataFile = os.path.expandvars(optionValue)
#         elif optionName == 'targetFolder':
#             targetFolder = os.path.expandvars(optionValue)
#         elif optionName == 'dsName':
#             dsName = optionValue
#         elif optionName == 'maxcount':
#             maxcount = int(float(optionValue))
#         elif optionName == 'maxsize':
#             maxsize = int(float(optionValue))
#         elif optionName == 'inMemory':
#             inMemory = optionValue == 'True'         
#         elif optionName == 'fileRegexp':
#             filePattern = re.compile(optionValue)           
#         elif optionName == 'groundTruthBaseName':
#             groundTruthBaseName = optionValue   
#             if groundTruthBaseName == "":
#                 groundTruthBaseName = False;
#
#     return trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, inMemory, filePattern, groundTruthBaseName       
#

def main(sysargs):
    args = parse_args(sysargs)
    ImageNetTools.buildShardsForDataset(args.trainDataFile, args.metaDataFile, args.targetFolder, args.datasetName, maxcount=args.maxcount, maxsize=args.maxsize, inMemory = args.inMemory, filePattern=args.filePattern, groundTruthBaseName=args.groundTruthBaseName)     

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit()