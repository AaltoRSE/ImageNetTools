'''
Created on Sep 29, 2021

@author: thomas
'''

import ImageNetTools
import sys
import getopt
import os
import re


def parseOptions(configFile):
    f = open(os.path.expandvars(configFile),'r');
    configOptions = f.readlines();
    f.close()
    dsName = 'Dataset'
    maxcount=100000
    maxsize=3e9
    inMemory = False
    filePattern = ImageNetTools.imageNetMapper.finalFilePattern
    groundTruthBaseName = False
    for option in configOptions:
        #Remove comments
        if '#' in option:
            option = option.split('#',1)[0]
        #Skip lines without aktual assignments
        if not '=' in option:
            continue
        cOption = option.split('=',1)
        #Remove leading and trailing whitespaces.
        optionName = cOption[0].lstrip().rstrip();
        optionValue = cOption[1].lstrip().rstrip();
        if optionName == 'dataFile':  
            trainDataFile = os.path.expandvars(optionValue);
        elif optionName == 'metaDataFile':
            metaDataFile = os.path.expandvars(optionValue)
        elif optionName == 'targetFolder':
            targetFolder = os.path.expandvars(optionValue)
        elif optionName == 'dsName':
            dsName = optionValue
        elif optionName == 'maxcount':
            maxcount = int(float(optionValue))
        elif optionName == 'maxsize':
            maxsize = int(float(optionValue))
        elif optionName == 'inMemory':
            inMemory = optionValue == 'True'         
        elif optionName == 'fileRegexp':
            filePattern = re.compile(optionValue)           
        elif optionName == 'groundTruthBaseName':
            groundTruthBaseName = optionValue   
                         
    return trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, inMemory, filePattern, groundTruthBaseName       
        
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:",["conf="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--conf"):
            try:
                trainDataFile, metaDataFile, targetFolder, dsName, maxcount, maxsize, inMemory, filePattern, groundTruthBaseName = parseOptions(arg);
            except:
                printHelp()
                sys.exit(2)
    ImageNetTools.buildShardsForTrainingDataset(trainDataFile, metaDataFile, targetFolder, dsName, maxcount=maxcount, maxsize=maxsize, inMemory = inMemory, filePattern=filePattern, groundTruthBaseName=groundTruthBaseName)     
    sys.exit()
   
def printHelp():
    print('Call by providing a config file:')
    print('python shard.py -c configFileName' )
    print('The options available in the config file are:')
    print('required:')
    print('dataFile=/path/to/data/file          # The path to the dataset tar file to shard')
    print('targetFolder=/path/to/output/folder  # The output folder for the shards')      
    print('metaDataFile=/path/to/meta/info/file # The path to the metaData file')
    print('Optional (with default values:')    
    print('fileRegexp=.*?([^/]+)/[^/]*\..*      # The regular expression to extract the class from the files')
    print("                                     # The default uses the folder name before the actual file")
    print('dsName=Dataset                       # The Base name of the output shards')
    print('maxcount=100000                      # Maximium number of files per shard')
    print('maxsize=3e9                          # Maximium size per shard')
    print('inMemory=False                       # Whether sharding should be performed entirely in memory')
    print('groundTruthBaseName=ILSVRC2012_val_  # The base name for a ground truth file to add the ID to (normally validation)')  
    
    
    
main(sys.argv[1:])    
    
    