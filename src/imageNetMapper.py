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

imagenet21k_resize_pattern = re.compile('.+/.+/(.*)\.JPEG')

def createInstancetoClassFromSynsetInfo(imageNetMetaFile):
    mapfile = sciio.loadmat(imageNetMetaFile)
    sets = mapfile['synsets']
    idmap = {};
    # get the mapping between WNID and ImageNetIDs        
    for x,y in zip(sets['WNID'],sets['ILSVRC2012_ID']):
        # ugly since the import packaes matlab data into multiple arrays
        idmap[x[0][0]] = y[0][0][0]
        
    return idmap
    


    
     
def buildShardsFromFolder(fileFolder, fileToClass, targetFolder, outputFileName, maxShardFiles=10000, filePattern = None):
    '''
    Build shards from a folder with image files and a given translation table between images and associated classes.
    
    Parameters
    fileFolder:         The file folder where the image files are located. 
    fileToClass:        The translation table how to get from the fileName to the Associated class
    targetFolder:       The folder to write the Shards to.
    outputFileName:     The Base name of the output file (ShardNumber and .tar will be added
    
    Optional Parameters:
    maxShardFiles       Maximum number of Files within one shard (default 10000)
    filePattern:        If non-empty the full relative path from the FileFolder to the images will be matched to 
                        this expression and the first matching group will be used to look up the Class. 
    '''
    
    res = []
    Files = glob.glob(os.path.join(fileFolder, '**')) #get all contents of the imageNet File Folder
    if filePattern == None:
        res = [(fname,fname) for fname in Files]
    else:             
        res = [(fname,re.match(fname)) for fname in Files]
                
    print(res)
    perm = np.random.permutation(len(res))
    numFileLength = str(ceil(log10(len(perm))))
    outputpattern = outputFileName + "%0" + numFileLength + "d.tar"
    with SW(os.path.join(targetFolder, outputpattern),maxcount=maxShardFiles) as writer:        
        # due to matching we can have entries.
        for i in perm:
            data = res[i]
            if data[1] != None:
                file = data[0]                
                fileclass = fileToClass[data[1]]
                key = os.path.splitext(file)[0]
                with open(file,'rb') as stream:
                    binary_data = stream.read()
                sample = {"__key__": key,
                          "jpg": binary_data,
                          "cls": fileclass}
                writer.write(sample)
                                
    
class ImageNetMapper(object):
    '''
    Offers a couple utility functions for imageNet datasets 
    '''

    

    def __init__(self, params):
        '''
        Constructor
        '''
        