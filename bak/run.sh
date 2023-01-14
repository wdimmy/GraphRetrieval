#!/bin/bash

# training base models 
ogb_dataset="ogbg-molbace ogbg-molbbbp ogbg-molclintox ogbg-molmuv ogbg-molsider ogbg-moltox21 ogbg-moltoxcast ogbg-molhiv"
lsc_dataset="PCQM4M PCQM4Mv2"
image_dataset="MNIST CIFAR10"


# LSC datasets needs lots of sources
gpuid=0
for dataset in $lsc_dataset
do
    nohup ./launch_gcn.sh $gpuid ogb $dataset 0 0 &
    gpuid=$((gpuid+1))
    nohup ./launch_gin.sh $gpuid ogb $dataset 0 0 &
    gpuid=$((gpuid+1))
    nohup ./launch_pna.sh $gpuid ogb $dataset 0 0 &
    gpuid=$((gpuid+1))
done 

for dataset in $ogb_dataset
do
    nohup ./launch_gcn.sh $gpuid ogb $dataset 0 0 &
    nohup ./launch_gin.sh $gpuid ogb $dataset 0 0 &
    nohup ./launch_pna.sh $gpuid ogb $dataset 0 0 &
done 

for dataset in $image_dataset
do
    nohup ./launch_gcn.sh $gpuid ogb $dataset 0 0 &
    nohup ./launch_gin.sh $gpuid ogb $dataset 0 0 &
    nohup ./launch_pna.sh $gpuid ogb $dataset 0 0 &
done 

wait
echo "Training for base models is done"


gpuid=0
for dataset in $lsc_dataset
do
    for k in `seq 1 3`
    do 
        nohup ./launch_gcn.sh $gpuid ogb $dataset 1 $k &
        nohup ./launch_gin.sh $gpuid ogb $dataset 1 $k &
        gpuid=$((gpuid+1))
        nohup ./launch_pna.sh $gpuid ogb $dataset 1 $k &
    done 
done 

for dataset in $ogb_dataset
do
    for k in `seq 1 3`
    do 
        nohup ./launch_gcn.sh $gpuid ogb $dataset 1 $k &
        nohup ./launch_gin.sh $gpuid ogb $dataset 1 $k &
        nohup ./launch_pna.sh $gpuid ogb $dataset 1 $k &
    done 
done 

for dataset in $image_dataset
do
    for k in `seq 1 3`
    do 
        nohup ./launch_gcn.sh $gpuid ogb $dataset 1 $k &
        nohup ./launch_gin.sh $gpuid ogb $dataset 1 $k &
        nohup ./launch_pna.sh $gpuid ogb $dataset 1 $k &
    done 
done 