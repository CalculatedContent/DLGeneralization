#!/usr/bin/env python
from random import randint
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Inputs for MLP3 variants.')
parser.add_argument('--random', metavar='r', type=int, default=0, help='% labels randomized')

args = parser.parse_args()

for i in range(10000):
    label = -1
    if args.random > 0:
        if randint(0,100)< args.random:
            label = np.identity(10)[randint(0, 9)]
            
    print(label)



            

