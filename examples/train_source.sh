#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s A -t A -a resnet50  --epochs 50 --seed 0 > benchmarks/train_source/Office31_A.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s D -t D -a resnet50  --epochs 50 --seed 0 > benchmarks/train_source/Office31_D.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s W -t W -a resnet50  --epochs 50 --seed 0 > benchmarks/train_source/Office31_W.txt

# Office-Home
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Ar -t Ar -a resnet50 --epochs 50 --seed 0 > benchmarks/train_source/OfficeHome_Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Cl -t Cl -a resnet50 --epochs 50 --seed 0 > benchmarks/train_source/OfficeHome_Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Pr -t Rr -a resnet50 --epochs 50 --seed 0 > benchmarks/train_source/OfficeHome_Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Rw -t Rw -a resnet50 --epochs 50 --seed 0 > benchmarks/train_source/OfficeHome_Rw.txt

# # # VisDA-2017
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/visda-2017 -d VisDA2017 -s T -t T -a resnet101 --epochs 10 --print-freq 1000 --lr 0.001 --seed 0 > benchmarks/train_source/VisDA2017_resnet101.txt
