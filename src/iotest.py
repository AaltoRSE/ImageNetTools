'''
Created on Sep 29, 2021

@author: thomas
'''

from . import ImageNetTools
import sys
import getopt
                        
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd:",["dataset="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):            
            ImageNetTools.benchmarkIOSpeeds(arg)     
    sys.exit()
   
def printHelp():
    print('Run IO Speed testing with a given Dataset')
    print('python iotest.py -d /path/to/dataset' )        
    
main(sys.argv[1:])