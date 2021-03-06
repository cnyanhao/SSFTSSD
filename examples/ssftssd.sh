# #!/usr/bin/env bash
# # Office31
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s A -t W -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_A2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s D -t W -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_D2W.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s W -t D -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_W2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s A -t D -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_A2D.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s D -t A -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_D2A.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office31 -d Office31 -s W -t A -a resnet50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/Office31_W2A.txt

# Office-Home
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Ar2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Ar2Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Ar2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Cl2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Cl2Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Cl2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Pr2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Pr2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Pr2Rw.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Rw2Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Rw2Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 50 --seed 2021 --uda_type cdan --param_cdan 1. > benchmarks/ssftssd/OfficeHome_Rw2Pr.txt

# VisDA-2017
# CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet50 --epochs 30 --seed 2021 > benchmarks/ssftssd/VisDA2017.txt
CUDA_VISIBLE_DEVICES=0 python examples/ssftssd.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet101 --print-freq 500 --epochs 15 --num_eval 15 --seed 2021 --uda_type cdan --param_cdan 0.1 --center_crop --fix_head > benchmarks/ssftssd/VisDA2017_resnet101.txt
