# NeLLCom-Lex 
## A Neural-agent Framework to Study the Interplay between Lexical Systems and Language Use

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

NeLLCom-Lex expands the scope to simulate the evolution of lexical meaning, while [NeLLCom](https://github.com/Yuchen-Lian/NeLLCom) focused on the emergence of universal word order properties.
NeLLCom-Lex agents communicate about a simplified referential world using pre-defined lexicons acquired during the supervised learning phase. 

The implementation of NeLLCom-Lex is partly based on [EGG](https://github.com/facebookresearch/EGG) toolkit.

More details can be found in [arxiv](xxx)


## Agent Architecture

Both speaking and listening agents are composed of feedforward neural networks (FNNs), following the common architecture design in referential communication games.


## Installing NeLLCom-Lex

1. Cloning NeLLCom-Lex:
   ```
   git clone git@github.com:yuqing0304/NeLLCom-Lex.git
   cd NeLLCom-Lex
   ```
4. Then, we can run a game, e.g. the color naming experiments conducted in the paper:
    ```bash
    bash experiment1.sh
    bash experiment2a.sh
    bash experiment2b.sh
    bash experiment2c.sh
    ```

## NeLLCom-Lex structure

* `data/` contains the full dataset of the color and their names that are used in the paper.
* `train.py` contain the actual logic implementation.
* ...


## Citation
If you find NeLLCom-Lex useful in your research, please considering to cite:
```
xxx
```
