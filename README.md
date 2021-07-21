# Ranking-preserved Temporal Graph Convolutional Network

This repository contains the python implementation for paper "A Gospel for MOBA Game: Ranking-Preserved Hero Change Prediction in Dota 2."

## Paper Abstract

Dota 2 is one of the most popular Multiplayer Online Battle Arena (MOBA) games, in which players of different teams controlling heroes fight each other to pursue a championship. To enrich the diversity of heroes and provide a balanced battle environment, Dota 2 game designers keep changing the attributes or skills of heroes in constantly updated game versions. Since Dota 2 is an intricate game, and numerous factors are involved in a match, it is challenging to figure out how to adjust heroes to meet the balance. As far as we know, no effort from intelligent learning perspective has been made to judge whether a hero should be changed in the next game version. This paper proposes a ranking-preserved method to predict whether a hero should be enhanced, weakened, or unchanged. Specifically, a unified Ranking-preserved Temporal Graph Convolutional Network (RT-GCN) model containing ranking preservation, GCN, and LSTM is designed to generate a ranking list of all the heroes, which indicates the strength of heroes as well as which heroes to be enhanced or weakened. The experiments on match records show that our approach provides a high-quality prediction and performs better than other baseline models. For game players, our model can give them a view of which heroes are more powerful currently and help them make better choices in choosing heroes. For game designers, our model can provide statistical supports and model interpretation for hero adjustment.

## Requirements

* Tensorflow 1.4
* numpy
* pandas

## File Description

* `data.py`: data pre-processing and data loading
* `DNN_rank_train.py`: training and test for DNN (DNN_rank) model and R (R_rank_parameter) model
* `model.py`: tensorflow implementation of models
* `gcn_rank_train.py`: training and test for R (R_rank_parameter) model, GCN (G_rank_parameter) model and G+R (RG_rank_parameter) model
* `r3gl_rank_train.py`: training and test for all other models (L_rank_parameter, GL_Rank_parameter, RL_Rank_parameter, R3GL_Rank_parameter)

## Datasets

* Collected by [OpenDota](https://www.opendota.com/)
* From version 7.00 to version 7.21d

## How to Run

For our method, change the model in `r3gl_rank_train.py` to 'R3GL_Rank_parameter' and then run
```bash
python r3gl_rank_train.py
```

To select the adjacency matrices to use, change the hyperparameter 'NUM_ADJ' in `r3gl_rank_train.py` and line 74, line 78, line 82 to the corresponding matrices, then run
```bash
python r3gl_rank_train.py
```

For baseline models, run the corresponding file as described in Section 'File Description.'
