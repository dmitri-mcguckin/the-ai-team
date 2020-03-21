#!/usr/bin/env python3
import sys

def usage():
    print("usage: ./run.py <mode (tsne | pdr)> <*kw args>")
    sys.exit(-1)

def main(args):
    if(len(args) == 0): usage()

    if(args[0] == "tsne"):
        from src import tsne
        tsne.start(args[1:])
    elif(args[0] == "pdr"):
        from src import pdr
    else: usage()

if __name__ == "__main__": main(sys.argv[1:])
