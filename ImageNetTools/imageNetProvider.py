'''
Created on Sep 7, 2021

@author: Thomas Pfau
'''
from multiprocessing import Process, Queue
from torch.utils.data import IterableDataset
from imageNetMemory import imageNetMemory
from operator import itemgetter

def startImageMemory(pushQueue, waitQueue, imageNetFiles, batchSize=1000, labels = []):
    '''
    Initialize and start a ImageNetMemory that reads in an imageNet file
    Parameters:
    pushQueue:         The Queue, where items from the dataset are pushed to.
    waitQueue:         The Queue the Memory will look for signals before reading on
    imageNetFiles:     One or more File-names where the ImageNet Dataset is stored (assumed to be tar files)
    batchSize:         The size of the requested batches
    labels:            The labels
    '''
    memory = imageNetMemory(pushQueue, waitQueue, imageNetFiles, batchSize, labels)
    memory.start()
    
    
class imageNetProvider(IterableDataset):
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
        self.finished = False
        
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        get a batch of inputs and outputs from the imageDataSet read by this Provider (blocks till a batch is available)
        
        Returns
        inputs,outputs : The inputs and outputs as provided by a torch DataLoader from the specifed set.
        
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
            
                                        
    def __del__(self):        
        # kill the remaining processes and deleted the queues
        del(self.q_get)
        del(self.q_push)              
        self.p.kill()
        
            

    
    
