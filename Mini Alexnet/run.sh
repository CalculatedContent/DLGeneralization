#!/usr/bin/env bash

source activate py27

# 100 epochs
#./cifar10_alexnet.py --save True long_run True  > cifar10_alexnet.out
./cifar10_alexnet.py --save True long_run True --regularize True  > cifar10_alexnet.wd.out
#./cifar10_alexnet.py --save True long_run True --random 100 > cifar10_alexnet.r100.out
./cifar10_alexnet.py --save True long_run True --random 100 --regularize True > cifar10_alexnet.r100.wd.out

# early stopping
for id in `seq 1 10`
do
    ./cifar10_alexnet.py --id $id  --regularize True > cifar10_alexnet.id$id.wd.out
    ./cifar10_alexnet.py --id $id  --random 100 > cifar10_alexnet.id$id.r100.out
    for bs in 2 16 32 50 100 150 250 500 1000
    do
	./cifar10_alexnet.py --id $id --batch_size $bs > cifar10_alexnet.id$id.b$bs.out
    done
done


