#!/usr/bin/env bash

source activate py27
./cifar10_alexnet.py  > cifar10_alexnet.out
./cifar10_alexnet.py --random 100 > cifar10_alexnet.r100.out


./cifar10_alexnet.py --batch_size 2   > cifar10_alexnet.b2.out
./cifar10_alexnet.py --batch_size 2  --random 100 > cifar10_alexnet.b2.r100.out
