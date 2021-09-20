'''
Created on Sep 7, 2021

@author: Thomas Pfau
'''
from multiprocessing import Process, Queue
import webdataset as wds
from torch.utils.data import DataLoader
import time
import os.path as path 
import torch.utils.data
from ImageNetMemory import startImageMemory
from operator import itemgetter

class imageNetProvider(torch.utils.data.IterableDataset):
    '''
    A class that provides an interface 
    '''            
    def __init__(self, imageNetFiles, batchSize, classes = []):
        '''
        Build a provider, starting to read the given imageNetFile/URL in a spawned process.
        The process will only be closed once the final batch has been obtained.
         
        Parameters
        imageNetFile : The URL/FileName of the imageNetFile to read.
        batchSize : The batchSize to be obtained (will be passed to DataLoader)
        numWorker : The number of workers the Dataloader loading the datatset should use (default 1)
        '''        
        
        self.q_get = Queue()
        self.q_push = Queue()
        self.classes = classes
        self.p = Process(target=startImageMemory, args=(self.q_get,self.q_push,imageNetFiles,batchSize, classes))
        self.p.start()  
        
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        get a batch of inputs and outputs from the imageDataSet read by this Provider (blocks till a batch is available)
        
        Returns
        inputs,outputs : The inputs and outputs as provided by a torch DataLoader from the specifed set.
        
        '''               
        #We'll have to wait for something to be in the queue
        batch = self.q_get.get(block=True)
        self.q_push.put(True)
        #Check, whether we received the termination signal
        if not batch:
            self.p.join()
            #Can't iterate any longer
            raise StopIteration 
        if self.classes.empty:       
            return batch    
        else:
            result = itemgetter(self.classes,batch)
            print(result)
            return result
            
                                        
    def __del__(self):        
        # in case we have not fetched all batches, still finish the reading process and clear the queues        
        while not self.q_get.empty():
            self.q_get.get()
        while not self.q_push.empty():
            self.q_push.get()            
        self.p.kill()
        
            

    
    
