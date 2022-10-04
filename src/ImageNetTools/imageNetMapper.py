'''
Created on Sep 10, 2021

@author: thomas
'''

import os 
import re
import pathlib
from math import ceil, log10
import tempfile
import json

import scipy.io as sciio
import numpy as np

import tarfile
from io import BytesIO as BinaryReader
import webdataset as wds
from webdataset.writer import ShardWriter as SW


finalFilePattern = '.*?([^/]+)/[^/]*\..*'

def getMatch(fileName, pattern):
    res = pattern.match(fileName)
    if res == None:
        return None
    else:
        return res.groups()[0]  
        
        
def buildShardsFromFolder(fileFolder, fileToClass, targetFolder, outputFileName, filePattern = None, maxcount=100000, maxsize=3e9, preprocess = None, dataType = "img", mappingFile="FileInfo.json", seed=1):
    '''
    Build shards from a folder with image files and a given translation table between images and associated classes.
    
    Parameters
    fileFolder:         The file folder where the image files are located. 
    fileToClass:        The translation table how to get from the fileName to the Associated class
    targetFolder:       The folder to write the Shards to.
    outputFileName:     The Base name of the output file (ShardNumber and .tar will be added
    
    Optional Parameters:
    maxcount:           Maximum number of files within one shard (default 100000)
    maxsize:            Maximum size of each shard(default 3e9)
    filePattern:        If non-empty the full relative path from the FileFolder to the images will be matched to 
                        this expression and the first matching group will be used to look up the Class.
    preprocess:         A function that takes in an read file and preprocesses the raw data. 
                        NOTE: The data provided to preprocess, is the raw data, if it's an image and you need an image object, 
                              you have to decode it in the preprocess function.
    dataType:           The type of data processed, will be the key, the data is stored under in the shards.
    '''
    rng = np.random.default_rng(seed)
    res = []     
    Files = [f for f in pathlib.Path(fileFolder).rglob('*.*')]    
    if filePattern == None:
        res = [(os.fspath(fname), os.path.splitext(os.fspath(fname.relative_to(fileFolder)))[0], os.fspath(fname.relative_to(fileFolder))) for fname in Files]
    else:             
        filePattern = re.compile(filePattern);
        res = [(os.fspath(fname), os.path.splitext(os.fspath(fname.relative_to(fileFolder)))[0], getMatch(os.fspath(fname.relative_to(fileFolder)),filePattern)) for fname in Files]
                
    #get an appropriate length of Shard Names
    perm = rng.permutation(len(res))
    numFileLength = ceil(log10(len(perm)/maxcount))
    if numFileLength < 1:
        numFileLength = 1
    numFileLength = str(numFileLength)
    outputpattern = outputFileName + "%0" + numFileLength + "d.tar"
    outputMetaData = {}
    with SW(os.path.join(targetFolder, outputpattern),maxcount=maxcount,maxsize=maxsize) as writer:        
        # due to matching we can have entries.
        for i in perm:
            data = res[i]            
            if data[1] != None:                         
                with open(data[0],'rb') as stream:
                    binary_data =stream.read()        
                sample = getSample(data[1], fileToClass[data[2]], preprocess, binary_data, dataType)                                 
                position,filename = writeSample(sample,writer)
                outputMetaData[data[0]] = {'class' : fileToClass[data[2]], 'targetFile' : filename.replace(targetFolder,''), 'position' : position}                
    with open(os.path.join(targetFolder,mappingFile), 'w') as metaFile:
        metaFile.write(json.dumps(outputMetaData))


def writeSample(sample, writer):
    '''
    Write a Sample, and obtain the filename and the position of the file in the written shard.
    '''
    oldfile = writer.fname
    oldpos = writer.size
    writer.write(sample)    
    if(writer.fname == oldfile):
        return oldpos, writer.fname
    else:
        return 0,writer.fname
    
    

def buildShardsFromSource(Files, fileToClass, targetFolder, outputFileName, filePattern = None, maxcount=100000, maxsize=3e9, preprocess = None, dataType = "img", mappingFile="FileInfo.json", seed = 1):
    '''
    Build shuffled shards from a source tar file keeping all elements in memory. 
    This function can easily fail if insufficient memory is allocated. 
    
    Parameters
    Files:              The in-memory dictionary containing fileName : image data pairs
    fileToClass:        The translation table how to get from the fileName to the Associated class
    targetFolder:       The folder to write the Shards to.
    outputFileName:     The Base name of the output file (ShardNumber and .tar will be added
    
    Optional Parameters:
    maxcount:           Maximum number of files within one shard (default 100000)
    maxsize:            Maximum size of each shard(default 3e9)
    filePattern:        If non-empty the full relative path from the FileFolder to the images will be matched to 
                        this expression and the first matching group will be used to look up the Class.
    preprocess:         A function that takes in an read file and preprocesses the raw data. 
                        NOTE: The data provided to preprocess, is the raw data, if it's an image and you need an image object, 
                              you have to decode it in the preprocess function.
    dataType:           The type of data processed, will be the key, the data is stored under in the shards.                              
    '''
    rng = np.random.default_rng(seed)
    res = []
    #Open the tarfile as stream.     
    if filePattern == None:
        res = [(fname,os.path.splitext(fname)[0], fname) for fname in Files]
    else:             
        filePattern = re.compile(filePattern);
        res = [(fname, os.path.splitext(fname)[0], getMatch(fname,filePattern)) for fname in Files]            
    #get an appropriate length of Shard Names
    perm = rng.permutation(len(res))
    numFileLength = str(ceil(log10(len(perm)/maxcount)))
    outputpattern = outputFileName + "%0" + numFileLength + "d.tar"
    outputMetaData = {}
    with SW(os.path.join(targetFolder, outputpattern),maxcount=maxcount,maxsize=maxsize) as writer:        
        # due to matching we can have entries.
        for i in perm:
            data = res[i]            
            if data[1] != None:
                binary_data = Files[data[0]]                         
                sample = getSample(data[1], fileToClass[data[2]], preprocess, binary_data, dataType)                    
                position,filename = writeSample(sample,writer)
                outputMetaData[data[0]] = {'class' : fileToClass[data[2]], 'targetFile' : filename.replace(targetFolder,''), 'position' : position}                
    with open(os.path.join(targetFolder,mappingFile), 'w') as metaFile:
        metaFile.write(json.dumps(outputMetaData))
                                
                                
def getSample(key, keyClass, preprocess, binary_data, dataType):        
    # Take only the base file name, the type and class will be added by the shardwriter
    print(key)
    if not preprocess == None:        
        binary_data = preprocess(binary_data)         
    sample = {"__key__": key,
                  dataType : binary_data,
                  "cls": str(keyClass)} # Metadata must be of type strring according to webdataset definitions
    return sample
    
    
class ImageNetMapper(object):
    '''
    Offers a couple utility functions for imageNet datasets 
    '''

    

    def __init__(self):
        '''
        Constructor
        '''
        self.idmap = {};
    
    def createInstanceToClassFromSynsetInfo(self,imageNetMetaFile):
        '''
        create Instance to class mapping from a meta.mat file containing the mapping between 
        WNIDs and imagenet classes. (for imagenet), or using a json file with "ID" : class otherwise.
        
        Parameters:
        imageNetMetaFile:   The meta.mat file from imagenet containing the synsets struct with ILSVRC2021_ID and WNID fields.
        '''
        if(imageNetMetaFile.endswith(".mat")):
            mapfile = sciio.loadmat(imageNetMetaFile)
            sets = mapfile['synsets']
            self.idmap = {};
            # get the mapping between WNID and ImageNetIDs        
            for x,y in zip(sets['WNID'],sets['ILSVRC2012_ID']):
                # ugly since the import packs matlab data into multiple arrays
                self.idmap[x[0][0]] = y[0][0][0]
        else:
            with open(imageNetMetaFile, 'r') as f:
                self.idmap = json.load(f)
            
    
    def shardDataFolder(self, dataFolder, metaDataFile,targetFolder, dsName,maxcount=100000, maxsize=3e9, preprocess = None, filePattern=finalFilePattern, groundTruthBaseName=False, dataType = "img", seed=1):
        '''
        Read in data from a folder, using either a metaDataFile 
        and build randomized shards from it, including labels.
        
        Parameters:
        dataFolder:         The location of the data files
        metaDataFile:       The location of the metaData .mat file to build the mapping
        targetFolder:       The output folder (optimally network space)
        dsName:             Base name of the output ffiles (The resulting sharded DS will be stored as dsname000X..XXXX.tar)
               
        Optional Parameters:
        maxcount:           Maximum number of files within one shard (default 100000)
        maxsize:            Maximum size of each shard(default 3e9)
        preprocess:         A function that takes in an read file and preprocesses the raw data. 
                            NOTE: The data provided to preprocess, is the raw data, if it's an image and you need an image object, 
                            you have to decode it in the preprocess function.
        filePattern:        The pattern used to extract the WNIDs for each element 
        dataType:           The type of data processed, will be the key, the data is stored under in the shards.
        seed:               The seed to generate the permutation (default: 1)
        '''
        if not groundTruthBaseName:
            self.createInstanceToClassFromSynsetInfo(metaDataFile)
        else:
            self.createInstanceToClassFromGroundTruth(metaDataFile, groundTruthBaseName)
            #No pattern, since we use ground-truthes.            
            filePattern = re.compile('.*?[^/]*?/?([^/]*\..*)')
        
        buildShardsFromFolder(dataFolder, self.idmap, targetFolder, dsName, filePattern=filePattern, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, dataType = dataType, seed=seed)
        
        
        
    def extractAndPackData(self, trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, filePattern=finalFilePattern, groundTruthBaseName=False, dataType='img', seed=1):
        '''
        Extract a Training data file (assumed to have the following internal structure:
        Train.tar 
        |--> class1.tar
        |--> class2.tar
        |...
        |--> classXYZ.tar
        and build randomized shards from it, including labels.
        
        Parameters:
        trainDataFile:      The location of the training data file
        metaDataFile:       The location of the metaData .mat file to build the mapping
        targetFolder:       The output folder (optimally network space)
        dsName:             Base name of the output ffiles (The resulting sharded DS will be stored as dsname000X..XXXX.tar)        
               
        Optional Parameters:
        maxcount:           Maximum number of files within one shard (default 100000)
        maxsize:            Maximum size of each shard(default 3e9)
        preprocess:         A function that takes in an read file and preprocesses the raw data. 
                            NOTE: The data provided to preprocess, is the raw data, if it's an image and you need an image object, 
                            you have to decode it in the preprocess function.
        filePattern:        The pattern used to extract the WNIDs for each element 
        dataType:           The type of data (only relevant for the key the data is stored under in the shards)
        seed:               The seed to generate the permutation (default: 1)
        '''

        #Extract All files to the local tmp directory, placing them in a directory named after the internal .jar File        
        tmpDir = self.readData(trainDataFile,True)
        # build the mapping
        # ground Truth base name only needs to be provided, if it actually is a base name. Otherwise this is an indicator that its 
        if not groundTruthBaseName:
            self.createInstanceToClassFromSynsetInfo(metaDataFile)
        else:
            self.createInstanceToClassFromGroundTruth(metaDataFile, groundTruthBaseName)
            #No pattern, since we use ground-truthes.            
            filePattern = '.*?[^/]*?/?([^/]*\..*)'
        # now, Create classes with the mapping
        buildShardsFromFolder(tmpDir, self.idmap, targetFolder, dsName, filePattern=filePattern, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, dataType=dataType, seed=seed)        
            
        
    def extractAndPackDataInMemory(self, trainDataFile, metaDataFile, targetFolder, dsName, maxcount=100000, maxsize=3e9, preprocess = None, filePattern=finalFilePattern, metaIsSynset=True, groundTruthBaseName=False, dataType='img', seed=1):
        '''
        Extract a Training data file (assumed to have the following internal structure:
        Train.tar 
        |--> class1.tar
        |--> class2.tar
        |...
        |--> classXYZ.tar
        and build randomized shards from it, including labels. 
        IMPORTANT: This function will load the whole dataset to memory and not use any
                   local storage. While making it faster than using local drives, it
                   requires large amounts of memory.
        
        Parameters:
        trainDataFile:      The location of the training data file
        metaDataFile:       The location of the metaData .mat file to build the mapping
        targetFolder:       The output folder (optimally network space)
        dsName:             Base name of the output ffiles (The resulting sharded DS will be stored as dsname000X..XXXX.tar)
               
        Optional Parameters:
        maxcount:           Maximum number of files within one shard (default 100000)
        maxsize:            Maximum size of each shard(default 3e9)
        preprocess:         A function that takes in an read file and preprocesses the raw data. 
                            NOTE: The data provided to preprocess, is the raw data, if it's an image and you need an image object, 
                            you have to decode it in the preprocess function.
        filePattern:        The pattern used to extract the WNIDs for each element  
        seed:               The seed to generate the permutation (default: 1)
        '''        
        Files = self.readData(trainDataFile,False)
        if not groundTruthBaseName:
            self.createInstanceToClassFromSynsetInfo(metaDataFile)
        else:
            self.createInstanceToClassFromGroundTruth(metaDataFile, groundTruthBaseName)
            #No pattern, since we use ground-truthes.            
            filePattern = None
            
        buildShardsFromSource(Files, self.idmap, targetFolder, dsName, filePattern=filePattern, maxcount=maxcount, maxsize=maxsize, preprocess=preprocess, dataType=dataType, seed=seed)        

        #Extract All files to the local tmp directory, placing them in a directory named after the internal .jar File        


    def readData(self, trainDataFile, toDisk):
        '''
        Read in the data from a training data set of imagemap. Training data is assumed to be in a 
        tar file, either as individual images, or again as tar files 
        
        Parameters:
        trainDataFile:     The tarballs containing the training data
        toDisk:            Whether to store the data to disk or in a dictonary in memory
        
        Returns:
        Files:             If toDisk is true, Files is the folder containing the data.
                           If toDisk sis false, Files is a dictionary of FilesName to Binary data
        '''
        
        if(toDisk):
            Files = tempfile.mkdtemp()
            print('Extracting individual files to : ' + Files, flush=True)            
        else:
            print("Reading Data to Memory", flush=True)
            Files = {}
        #this is more efficient than using tar files, and handles all our issues.
        
        if toDisk:
            # The tar-balls are pretty large, so we reduce the number of tarballs extracted at each time to a few.
            dataset = wds.WebDataset(trainDataFile,cache_size = 4)
        else:
            # We assume there is sufficient memory.
            dataset = wds.WebDataset(trainDataFile)
            
        for element in dataset:
            #Here, we will check, whether this is a tar of tar or a tar of JPEGs.            
            currentClassName = element['__key__']            
            #print("Current class: " + currentClassName , flush=True)
            #Create a directory for all those files.
            outFolder = currentClassName
            if 'tar' in element.keys():
                # this is a tar of tars. 
                f = BinaryReader(element['tar']);                
                innerTarFile = tarfile.open(fileobj=f)
                innerJPEG = innerTarFile.next()                                    
                #Process tar for the current Class
                if toDisk:                        
                    outFolder = os.path.join(Files, currentClassName);                                   
                    os.mkdir(outFolder)
                while not innerJPEG == None:
                    JPEGFile = innerTarFile.extractfile(innerJPEG)
                    binary_data = JPEGFile.read() 
                    self.storeData(binary_data,toDisk,FileName=os.path.join(outFolder,innerJPEG.name),MemoryDictionary=Files)
                    innerJPEG = innerTarFile.next()                                                        
            else:                
                if 'jpeg' in element.keys():
                    #only jpegs, directly store them.
                    fName = element['__key__']+'.JPEG' 
                    if toDisk:
                        fName = os.path.join(Files, fName)                    
                    binary_data = element['jpeg']           
                    self.storeData(binary_data,toDisk,FileName=fName,MemoryDictionary=Files)
        return Files
                
    def storeData(self,data,toDisk=False,FileName=None,MemoryDictionary={}):
        if toDisk:
            try:
                outfile = open(FileName,'wb')
            except FileNotFoundError:
                # potentially, we have to build the path
                dirToBuild = os.path.dirname(FileName)
                os.mkdir(dirToBuild)
                outfile = open(FileName,'wb')
                                
            outfile.write(data)
            outfile.close()
        else:
            MemoryDictionary[FileName] = data
        
        
    
    def createInstanceToClassFromGroundTruth(self, groundTruthFile, baseName):
        '''
        create Instance to class mapping from a ground truth file for a given 
        validation set assuming that the files have the form:
        baseName_XXXXXXXX.JPEG
        
        Parameters:
        groundTruthFile:    The ground truth (i.e. class) file in text format, with one line per image
        baseName:           The base name of the validation file.
        '''
        i = 1;
        groundTruth = open(groundTruthFile);
        cline = groundTruth.readline()
        while cline !='':
            cclass = cline.strip();
            self.idmap[baseName + "%08d" % i + ".JPEG"] = cclass
            cline = groundTruth.readline();
            i+=1      
        groundTruth.close()          
        
    def getTrainingPattern(self):
                
        return re.compile('.+/([^/]+)\.JPEG')
                
    def getIdmap(self):
        return self.idmap
