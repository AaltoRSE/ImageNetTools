'''
Created on Sep 10, 2021

@author: thomas
'''

import os 
import glob
import scipy.io as sciio
import webdataset as wds
from webdataset.writer import ShardWriter as SW
import re
import numpy as np
from math import ceil, log10
import tarfile
import tempfile

     
def getMatch(fileName, pattern):
    res = pattern.match(fileName)
    if res == None:
        return None
    else:
        return res.groups()[0]  
        
        
def buildShardsFromFolder(fileFolder, fileToClass, targetFolder, outputFileName, filePattern = None, maxcount=100000, maxsize=3e9, preprocess = None):
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
    '''
    
    res = []
    Files = glob.glob(os.path.join(fileFolder, '**')) #get all contents of the imageNet File Folder
    if filePattern == None:
        res = [(fname, fname) for fname in Files]
    else:             
        res = [(fname, getMatch(fname,filePattern)) for fname in Files]
                
    print(res)
    #get an appropriate length of Shard Names
    perm = np.random.permutation(len(res))
    numFileLength = str(ceil(log10(len(perm)/maxcount)))
    outputpattern = outputFileName + "%0" + numFileLength + "d.tar"
    with SW(os.path.join(targetFolder, outputpattern),maxcount=maxcount,maxsize=maxsize) as writer:        
        # due to matching we can have entries.
        for i in perm:
            data = res[i]
            if data[1] != None:
                file = data[0]                
                fileclass = fileToClass[data[1]]
                key = os.path.splitext(file)[0]
                with open(file,'rb') as stream:
                    binary_data = stream.read()
                    
                if not preprocess == None:
                    binary_data = preprocess(binary_data)
                    
                sample = {"__key__": key,
                          "jpg": binary_data,
                          "cls": fileclass}
                writer.write(sample)
                                
    
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
        WNIDs and imagenet classes.
        
        Parameters:
        imageNetMetaFile:   The meta.mat file from imagenet containing the synsets struct with ILSVRC2021_ID and WNID fields.
        '''
        mapfile = sciio.loadmat(imageNetMetaFile)
        sets = mapfile['synsets']
        self.idmap = {};
        # get the mapping between WNID and ImageNetIDs        
        for x,y in zip(sets['WNID'],sets['ILSVRC2012_ID']):
            # ugly since the import packs matlab data into multiple arrays
            self.idmap[x[0][0]] = y[0][0][0]
    
    def extractAndPackTrainData(self, TrainDataFile, metaDataFile, dsName):
        '''
        Extract a Training data file (assumed to have the following internal structure:
        Train.tar 
        |--> class1.tar
        |--> class2.tar
        |...
        |--> classXYZ.tar
        and build randomized shards from it, including labels.
        '''
        tmpDir = tempfile.mkdtemp()
        
        file = tarfile.open(TrainDataFile,mode='r|')
        currentfile = file.next()
        #Extract All files to the local tmp directory, placing them in a directory named after the internal .jar File        
        while not currentfile == None:
            currentClassName = os.path.splitext(currentfile.name)[0]
            innerFile = file.extractfile(currentfile)
            innerTarFile = tarfile.open(fileobj=innerFile,mode='r|')
            innerJPEG = innerTarFile.next()
            #Create a directory for all those files.
            outFolder = os.path.join(tmpDir, currentClassName);
            os.mkdir(outFolder)            
            while not innerJPEG == None:            
                JPEGFile = innerTarFile.extractfile(innerJPEG)
                outfile = open(os.path.join(outFolder,innerJPEG.name),'wb')
                outfile.write(JPEGFile.read())
                outfile.close()
                innerJPEG = innerTarFile.next()
            currentfile = file.next()
        # build the mapping
        self.createInstanceToClassFromSynsetInfo(metaDataFile)
        # now, Create classes with the mapping
        tmpOutFolder = tempfile.mkdtemp();
        buildShardsFromFolder(tmpDir,self.idmap,tmpOutFolder,dsName,)
    
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
            cclass = cline;
            self.idmap[baseName + "%08d" % i + ".JPEG"] = cclass
            cline = groundTruth.readline();
            i+=1                
        
    def getTrainingPattern(self):
                
        return re.compile('.+/.+/(.*)\.JPEG')
                
    def getIdmap(self):
        return self.idmap
