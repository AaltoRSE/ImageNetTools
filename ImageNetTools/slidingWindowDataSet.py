'''
Created on Oct 11, 2021

@author: thomas
'''

from torch.utils.data import Dataset
import webdataset as wds
from torch.utils.data import DataLoader





class BasicDataSet(Dataset):
    '''
    classdocs
    '''

    
    def __init__(self, sourceFile, timeID, keys=["cls","data"], preprocess = None, numWorker=4, batchSize=1000):
        '''
        Constructor using a time-indicator and a 
        Parameters:
        sourceFile:      The path to a .tar source file
        timeID:          The key storing the sliding window property
        keys:            The other keys this dataset will provide as return tuples (default: ["cls", "data"] for class and data elements.
        preprocess:      preprocessing step for all loaded data (default: None)
        numWorker:       Number of worker to read the data (default: 4)
        batchSize:       Batch Size to read (default: 1000)
        '''
        self.timeID = timeID
        self.keys = keys
        createdTuples = keys[:]
        createdTuples.append(timeID)        
        if preprocess == None:
            ds = wds.WebDataset(sourceFile).to_tuple(*createdTuples)
        else:
            #preprocess should only ever act on the classes defined by keys.
            ds = wds.WebDataset(sourceFile).to_tuple(*createdTuples).map_tuple(preprocess)
        # Now lets load the data
        dl = DataLoader(ds,batch_size=batchSize, num_workers=numWorker)
        self.datastruct = []
        initDatastruct = True
        for element in dl:
            if initDatastruct:
                #create a list of empty lists
                self.datastruct = [[] for i in range(len(element))]
                initDatastruct = False
            for i in range(len(element)-1):
                #extend the individual arrays, except for the timeID, which is handled separately
                self.datastruct[i].extend([*element[i]])
            #The sliding Window variable is a float (and hopefully this is sorted.
            self.datastruct[-1].extend([float(i) for i in element[-1]])    
        #ensure,that the timeID is correct and potentially update all elements
        timeIDpos = [self.datastruct[-1][i] for i in range(len(self.datastruct[-1]))]
        timeIDpos.sort()
        timeID,permutation = zip(*timeIDpos)
        for i in range(len(self.datastruct)-1):
            self.datastruct[i] = [self.datastruct[i][pos] for pos in permutation]
            
    def setSlidingWindow(self,startVal,stopVal):
        self.startVal = startVal
        self.stopVal = stopVal        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        Get the next tuple from the sliding window.
        
        Returns
        
        
        '''               
        #We'll have to wait for something to be in the queue
        if self.finished:
            raise StopIteration
        
        batch = self.q_get.get(block=True)
        self.q_push.put(True)
        #Check, whether we received the termination signal
        if not batch:
            self.p.join()
            #Can't iterate any longer
            self.finished = True
            raise StopIteration 
        if len(self.classes) == 0:       
            return batch    
        else:
            result = itemgetter(self.classes,batch)
            print(result)
            return result            
        
            
        
        