#!/usr/bin/env python3
import sys
import urllib.request
import gzip
from pathlib import Path
from glob import glob
import os
w = 128
if __name__ == "__main__":
    for folder in glob(os.path.join(".", "*", "")):
        folder = folder.replace("./","", 1)
        cmd = "python calEigen.py -f {} -w {}".format(folder,w)
        outfile = os.path.join(folder, "sorted_eigens_{}.txt".format(w))
        print (outfile)
        if not os.path.exists(outfile):
            os.system(cmd)
        else:
            print("Exists:", outfile)