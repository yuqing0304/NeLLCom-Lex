# NeLLCom-Lex 
### A Neural-agent Framework to Study the Interplay between Lexical Systems and Language Use

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

NeLLCom-Lex extends the original [NeLLCom](https://github.com/Yuchen-Lian/NeLLCom) framework by simulating the evolution of lexical meaning, whereas NeLLCom primarily focused on the emergence of universal word order properties. In NeLLCom-Lex, agents communicate within a simplified referential world using pre-defined lexicons acquired during a supervised learning phase.

The implementation of NeLLCom-Lex is partly based on [EGG](https://github.com/facebookresearch/EGG) toolkit.

More details can be found in [TODOArxiv](xxx)


## Agent Architecture

Both speaking and listening agents are composed of feedforward neural networks (FNNs), following the common architecture design in referential communication games.


## Installing NeLLCom-Lex

1. Cloning NeLLCom-Lex:
   ```
   git clone git@github.com:yuqing0304/NeLLCom-Lex.git
   cd NeLLCom-Lex/EGG/egg/zoo/color_game
   or
   cd NeLLCom-Lex/EGG/egg/zoo/color_gamezero
   ```
4. Then, we can run a game, e.g., the color naming experiments conducted in the paper:
    ```bash
    sbatch run.sh
    ```
5. For the analyses of all metrics presented in the paper: 
    ```bash
    sbatch experiment1.sh
    sbatch experiment2a.sh
    sbatch experiment2b.sh
    sbatch experiment2c.sh
    ```

## NeLLCom-Lex structure

* `data/` contains the full dataset of the colors and their names that are used in the paper.
* `train.py` contains the actual logic implementation.
* `models.py`, `datasets.py`, and `utils_condition.py` contain the models, datasets, and utility functions.


## Citation
If you find NeLLCom-Lex useful in your research, please consider citing:
```
xxx
```

We also encourage you to cite the foundational works that NeLLCom-Lex builds upon:
```
@inproceedings{gualdoni-boleda-2024-objects,
    title = "Why do objects have many names? A study on word informativeness in language use and lexical systems",
    author = "Gualdoni, Eleonora  and
      Boleda, Gemma",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1009/",
    doi = "10.18653/v1/2024.emnlp-main.1009",
    pages = "18150--18163"
}

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
```
