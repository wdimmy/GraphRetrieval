This repository contains code for ECAI 2023 paper **An Empirical Study of Retrieval-enhanced Graph Neural Networks** by Dingmin Wang, Shengchao Liu, Hanchen Wang, Bernardo Cuenca Grau,Linfeng Song, Jian Tang, Le Song, Qi Liu.

## Dependencies
- python 3.8
- numpy
- torch==1.11.0
- torch-geometric==2.1.0
- faiss-gpu
- tensorboard
- PyTDC
- ogb

**Notice**, we also provide the Dockerfile and we encourage interested users to run our codes using the Docker to avoid some annoying error debugging. 


# Data

### Download data
All datasets used in our paper are publicly available. 

####  USPTO_Catalyst
```python 
from tdc.multi_pred import Catalyst
from ogb.utils import smiles2graph
data = Catalyst(name='USPTO_Catalyst', path="./dataset")
```

#### PCQM4M
```python
from ogb.lsc import PygPCQM4MDataset
dataset = PygPCQM4MDataset(root = 'dataset/')
```

#### PCQM4Mv2
```python
from ogb.lsc import PygPCQM4Mv2Dataset
dataset = PygPCQM4Mv2Dataset(root = 'dataset/')
```

#### 8 Molecular Datasets
```python 
from ogb.graphproppred import PygGraphPropPredDataset
# [ogbg-molhiv, ogbg-molbace, ogbg-molbbbp, ogbg-molclintox, ogbg-molmuv, ogbg-molpcba, ogbg-molsider, ogbg-moltox21, ogbg-moltoxcast]
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv) 
```

## GraphRetrieval

GraphRetrieval contains two phase of training

### Train Baseline 

#### Single-instance datasets training 

[`train_gcn.py`](train_gcn.py) provides the code to train a GCN model  over single instance graph-structured datasets **without** retrieval enhancement. An example usage of the script is given below:

```shell
python train_gcn.py \
        --source lsc \
        --dataset PCQM4M \
        --output_model_dir save \
```
Users can try more arguments e.g., --batch_size, --num_layers, etc. 

#### Multi-instance datasets training 

[`train_ddi_gcn.py`](train_ddi_gcn.py) provides the code to train a GCN model over multi-instance graph-structured datasets **without** retrieval enhancement. An example usage of the script is given below:

```shell
python train_gcn.py \
        --source ddi \
        --dataset USPTO \
        --output_model_dir save \
```
Users can try more arguments e.g., --batch_size, --num_layers, etc. 

### Train Retrieval-enhanced models

#### Single-instance datasets training 

[`train_gcn.py`](train_gcn.py) also provides the code to train a retrieval-enhanced GCN model over single-instance graph-structured datasets. An example usage of the script is given below:

```shell
python train_gcn.py \
        --source lsc \
        --dataset PCQM4M \
        --output_model_dir save \
        --retrieval 1 \
        --k 3 \    
```
where **--retrieval 1** denotes the retrieval-enhanced training mode, in which the checkpoints of the corresponding baseline models must exist, otherwise, there
will be errors. **--k** means the number of retrieved examples.

#### Multi-instance datasets training 

[`train_ddi_gcn.py`](train_ddi_gcn.py) provides the code to train a retrieval-enhanced GCN model over multi-instance graph-structured datasets. An example usage of the script is given below:

```shell
python train_gcn.py \
        --source ddi \
        --dataset USPTO \
        --output_model_dir save \
        --retrieval 1 \
        --k 3 \ 
```
where **--retrieval 1** denotes  the retrieval-enhanced training mode, in which the checkpoints of the corresponding baseline models must exist, otherwise, there
will be errors. **--k** means the number of retrieved examples.

#### Citation:
If you find our paper and resources useful, please kindly cite our paper:

```bibtex
@incollection{graphretrieval,
  title={An Empirical Study of Retrieval-enhanced Graph Neural Networks},
  author={Dingmin Wang, Shengchao Liu, Hanchen Wang, Bernardo Cuenca Grau,Linfeng Song, Jian Tang, Le Song and Qi Liu},
  booktitle={ECAI 2023},
  year={2023},
}
```

#### Contact
If you have any questions, feel free to contact me via (dingmin.wang@cs.ox.ac.uk).
