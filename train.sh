#!/usr/bin/bash
#python train.py --save='work/lr/lr0.3' --lr=0.3 &&
python train.py --save='work/lr/lr0.1' --lr=0.1 &&
python train.py --save='work/lr/lr0.03' --lr=0.03 &&
python train.py --save='work/lr/lr0.01' --lr=0.01

