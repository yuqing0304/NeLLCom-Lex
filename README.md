# NeLLCom-Lex 
### A Neural-agent Framework to Study the Interplay between Lexical Systems and Language Use

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

NeLLCom-Lex expands the scope to simulate the evolution of lexical meaning, while [NeLLCom](https://github.com/Yuchen-Lian/NeLLCom) focused on the emergence of universal word order properties.
NeLLCom-Lex agents communicate about a simplified referential world using pre-defined lexicons acquired during the supervised learning phase. 

The implementation of NeLLCom-Lex is partly based on [EGG](https://github.com/facebookresearch/EGG) toolkit.

More details can be found in [Arxiv](xxx)


## Agent Architecture

Both speaking and listening agents are composed of feedforward neural networks (FNNs), following the common architecture design in referential communication games.


## Installing NeLLCom-Lex

1. Cloning NeLLCom-Lex:
   ```
   git clone git@github.com:yuqing0304/NeLLCom-Lex.git
   cd NeLLCom-Lex/EGG/egg/zoo/color_game
   or
   cd NeLLCom-Lex/EGG/egg/zoo/color_game_zero
   ```
4. Then, we can run a game, e.g., the color naming experiments conducted in the paper:
    ```bash
    bash run.sh
    ```
5. For the analysis: 
    ```bash
    bash experiment1.sh
    bash experiment2a.sh
    bash experiment2b.sh
    bash experiment2c.sh
    ```

## NeLLCom-Lex structure

* `data/` contains the full dataset of the colors and their names that are used in the paper.
* `train.py` contains the actual logic implementation.
* `models.py`, `datasets.py`, `utils_condition.py`contain the models, datasets, and utility functions.


## Citation
If you find NeLLCom-Lex useful in your research, please consider citing:
```
@misc{kharitonov:etal:2021,
  author = "Kharitonov, Eugene  and Dess{\`i}, Roberto and Chaabouni, Rahma  and Bouchacourt, Diane  and Baroni, Marco",
  title = "{EGG}: a toolkit for research on {E}mergence of lan{G}uage in {G}ames",
  howpublished = {\url{https://github.com/facebookresearch/EGG}},
  year = {2021}
}

@article{lian2023communication,
    author = {Lian, Yuchen and Bisazza, Arianna and Verhoef, Tessa},
    title = "{Communication Drives the Emergence of Language Universals in Neural Agents: Evidence from the Word-order/Case-marking Trade-off}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1033-1047},
    year = {2023},
    month = {08},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00587},
    url = {https://doi.org/10.1162/tacl\_a\_00587}
}

xxx
```
