
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('INPUT', type=str, help='Input directory')
parser.add_argument('--outdir', type=str, 
                    default='.', help='Output directory')

args = parser.parse_args()

for filename in sorted(glob.glob(os.path.join(args.INPUT,'*.json'))):

    try:
        with open(filename) as f:
            data = json.load(f)
    except:
    	print(filename)