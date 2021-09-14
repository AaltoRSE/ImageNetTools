'''
Created on Sep 7, 2021

@author: Thomas Pfau
'''
from multiprocessing import Process, Queue
import webdataset as wds
from torch.utils.data import DataLoader


def readImageNet(q1, q2, imageNetUrl, batchsize, numWorkers = 4):
    '''
    read data from an imageNet File and put batches of a defined size into a queue to be extracted elsewhere
    
    Parameters
    
    q1 : a multiprocessing queue to put batches of images in (multiprocessing.Queue)
    q1 : a multiprocessing queue obtain signals when batches are taken out (multiprocessing.Queue)
    imageNetUrl : a FileName/URL of an imageNet file (String)
    batchsize : the batchsize of the dataloader (int) 
    '''
    
    print('Starting to read file')
    print(imageNetUrl)
    queuesize = 0;
    dataset = wds.WebDataset(imageNetUrl).shuffle(1000).to_tuple("jpg;png","cls")
    dataloader = DataLoader(dataset,num_workers=numWorkers,batch_size=batchsize)     
    for inputs,outputs in dataloader:
        if queuesize > 10:
            # this is ugly, and I would prefer something better... 
            q2.get(block=True)
            queuesize-=1
        q1.put([inputs,outputs])
        queuesize+=1
        
    #Finally, if we can't read anything any more, we send a signal to close the Process and to close the queue.
    q1.put(False)    

class imageNetProvider(object):
    '''
    A class that provides an interface 
    '''            
    def __init__(self, imageNetFile, batchSize, numWorker = 1):
        '''
        Build a provider, starting to read the given imageNetFile/URL in a spawned process.
        The process will only be closed once the final batch has been obtained.
         
        Parameters
        imageNetFile : The URL/FileName of the imageNetFile to read.
        batchSize : The batchSize to be obtained (will be passed to DataLoader)
        numWorker : The number of workers the Dataloader loading the datatset should use (default 1)
        '''        
        
        self.q_get = Queue();
        self.q_push = Queue();
        self.p = Process(target=readImageNet, args=(self.q_get,self.q_push,imageNetFile,batchSize))
        self.p.start()  
                      
    def __del__(self):        
        # in case we have not fetched all batches, still finish the reading process and clear the queues        
        while not self.q_get.empty():
            self.q_get.get()
        while not self.q_push.empty():
            self.q_push.get()            
        self.p.kill()
        
            
    def getBatch(self):
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
            return None
        
        return batch[0],batch[1]
