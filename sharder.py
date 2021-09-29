'''
Created on Sep 29, 2021

@author: thomas
'''

import ImageNetTools
import sys
import getopt
from _ast import Try

def parseOptions(configFile):
    f = open(configFile,'r');
    configOptions = f.readlines();
    f.close()
    dsName = 'Dataset'
    maxcount=100000
    maxsize=3e9
    inMemory = False
    for option in configOptions:
        #Remove comments
        if '#' in option:
            option = option.split('#')[0]
        #Skip lines without aktual assignments
        if not '=' in option:
            continue
        cOption = option.split('=')
        #Remove leading and trailing whitespaces.
        optionName = cOption[0].lstrip().rstrip();
        optionValue = cOption[1].lstrip().rstrip();
        if optionName == 'dataFile':  
            trainDataFile = optionValue;
        elif optionName == 'metaDataFile':
            metaDataFile = optionValue
        elif optionName == 'targetFolder':
            targetFolder = optionValue
        elif optionName == 'dsName':
            dsName = optionValue
        elif optionName == 'maxcount':
            maxcount = int(float(optionValue))
        elif optionName == 'maxsize':
            maxsize = int(float(optionValue))
        elif optionName == 'inMemory':
            inMemory = optionValue == 'True'         
                     
    return trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, inMemory        
        
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:",["conf="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--conf"):
            try:
                trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, inMemory = parseOptions(arg);
            except:
                printHelp()
                sys.exit(2)
    ImageNetTools.buildShardsForTrainingDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, inMemory = False)     
    sys.exit()
   
def printHelp():
    print('Call by providing a config file:')
    print('python shard.py -c configFileName' )
    print('The options available in the config file are:')
    print('required:')
    print('dataFile=/path/to/data/file          # The path to the dataset file to shard')
    print('metaDataFile=/path/to/meta/data/file # The path to the metaData file')
    print('targetFolder=/path/to/output/folder  # The output folder for the shards')      
    print('Optional (with default values:')
    print('dsName=Dataset            # The Base name of the output shards')
    print('maxcount=100000           # Maximium number of files per shard')
    print('maxsize=3e9               # Maximium size per shard')
    print('inMemory=False            # Whether sharding should be performed entirely in memory')  
    
    
main(sys.argv[1:])    
    
    