'''
Created on Sep 20, 2021

@author: thomas
'''

from multiprocessing import Process, Queue
import webdataset as wds
from torch.utils.data import DataLoader

class ComChannel():
    def __init__(self, qIn, qOut, process):
        self.qIn = qIn
        self.qOut = qOut
        self.process = process
        self.retrievedElements = 0;
        
        

def readImageNet(q1, q2, imageNetUrl, batchsize, classes = ["__key__","jpg;png","cls"]):
    '''
    read data from an imageNet File and put batches of a defined size into a queue to be extracted elsewhere
    it puts triplets of keys / items / classes into the queue. If all data is read, it will add `False` to the queue 
    Parameters batches should have at most batchsize elements, but can have less, depending on the number of workers.
    
    q1 : a multiprocessing queue to put batches of images in (multiprocessing.Queue)
    q1 : a multiprocessing queue obtain signals when batches are taken out (multiprocessing.Queue)
    imageNetUrl : a FileName/URL of an imageNet file (String)
    batchsize : the batchsize of the dataloader (int)        
    '''
    
    print('Starting to read file')
    print(imageNetUrl)
    queuesize = 0;
    if len(classes) == 0:
        dataset = wds.WebDataset(imageNetUrl).shuffle(1000)
    else:
        dataset = wds.WebDataset(imageNetUrl).shuffle(1000).to_tuple(*classes)
            
    for sample in dataset:        
        if queuesize > batchsize:
            # this is ugly, and I would prefer something better...            
            retrievedEntries = q2.get(block=True)
            queuesize-=retrievedEntries
        q1.put(sample)                    
        queuesize+=1
        
    #Finally, if we can't read anything any more, we send a signal to close the Process and to close the queue.
    q1.put(False)    


def readImageNetBasic(q1, imageNetUrl):
    '''
    read data from an imageNet File and put it into the given queue
    q1 : a multiprocessing queue to put batches of images in (multiprocessing.Queue)
    q1 : a multiprocessing queue obtain signals when batches are taken out (multiprocessing.Queue)
    imageNetUrl : a FileName/URL of an imageNet file (String)        
    '''
    
    print('Starting to read file')
    print(imageNetUrl)
    dataset = wds.WebDataset(imageNetUrl)
    dataloader = DataLoader(dataset)     
    for element in dataloader:        
        q1.put(element)                
    #Finally, if we can't read anything any more, we send a signal to close the Process and to close the queue.
    q1.put(False)    


    

class imageNetMemory(object):
    '''
    classdocs
    '''


    def __init__(self, pushQueue, waitQueue, imageNetFiles, batchSize=1000, classes = ["__key__","jpg;png","cls"]):
        '''
        Constructor
        '''
        self.readerCount = 0;
        self.resultQueue = pushQueue        
        self.batchSize = batchSize
        self.classes = classes
        self.waitQueue = waitQueue
        self.pushedCount = 0
        self.comChannels = [];        
        print("Checking imageNet File")
        if type(imageNetFiles) == type("") or type(imageNetFiles) == type(''):
            imageNetFiles = [imageNetFiles]
        self.imageNetFiles = imageNetFiles
        print("Setting up imagenet Workers")                          
        for file in self.imageNetFiles:
            q_get = Queue();
            q_push = Queue();
            p = Process(target=readImageNet, args=(q_get,q_push,file,self.batchSize, self.classes))
            p.start()
            self.comChannels.append(ComChannel(q_get,q_push,p))        
            self.readerCount += 1;
            print("Started Worker number %01i", self.readerCount)
                            
    def __del__(self):
        for comChannel in self.comChannels:
            #kill all spawned processes, if this object is killed
            comChannel.process.kill()
    
    def start(self):
        while self.readerCount > 0:            
            chansToDel = []
            for comChannel in self.comChannels:
                if not comChannel.qIn.empty():
                    pushElement = comChannel.qIn.get();
                    if pushElement:
                        self.resultQueue.put(pushElement)
                        comChannel.retrievedElements += 1;
                    else:
                        self.readerCount-=1
                        chansToDel.append(comChannel)
            #clean up remaining processes
            for comChannel in chansToDel:#
                #End process
                comChannel.process.join()
                self.comChannels.remove(comChannel)
            if not self.waitQueue.empty():
                #clear the element
                self.waitQueue.get()                
                #If we have processed data, clear the processed data from all reading processes
                removed = self.batchSize;
                for comChannel in self.comChannels:
                    # free Com channels
                    c_removed = comChannel.retrievedElements;
                    if ((removed - c_removed) < 0):
                         toRemove  = removed 
                    else:
                         toRemove = c_removed 
                    removed = removed - c_removed;                                                        
                    comChannel.qOut.put(toRemove)                    
                    comChannel.retrievedElements = c_removed - toRemove;
        #Send the termination signal
        self.resultQueue.put(False)