# Retrieval-enhanced Graph Neural Networks for Graph Property Prediction

  [Retrieval-enhanced Graph Neural Networks for Graph Property Prediction](#retrieval-enhanced-graph-neural-networks-for-graph-property-prediction)
  - [Datasets](#datasets)
  - [Running Environment](#running-environment)
  - [Running the code](#running-the-code)


## Datasets
All datasets we use are publicly available. In particular, the 8 molecule datasets (BBBP,Tox21,ToxCast,SIDER,ClinTox,MUV,HIV,BACE) and the two large-scale quantum chemistry datasets (PCQM4M and PCQM4Mv2) can be obtained from [ogb](https://github.com/snap-stanford/ogb). Two image classification datasets are obtained from [torch_geometric](https://github.com/pyg-team/pytorch_geometric). All downloaded datasets will be kept in the `Datasets` folders.

## Running Environment
To avoid some unnecessary environment-related issues, we encourage to run our codes in a docker container. The `Dockerfile` has been provided, which listes all dependencies. 

## Running the code
We provide a bash file `run.sh` for the training and the testing. The trained models will be kept in `Save` folder in the format of `dataset-basemodel-k.pth`, where `datasetname` represents the dataset (e.g., BBBP or Tox21), `basemodel` dennotes the gnn
model (e.g., gin, gcn or pna) and `k` represents the number of retrieved results (in particular, 0 represents the model without using retrieval). All best testing results are kept in total.csv file, which also locates in the `Save` folder.
