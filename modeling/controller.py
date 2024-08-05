"""
Our implementation of MRNN model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""


import torch

class Controller:
    def __init__(self, early_stop_patience):
        self.early_stop_patience = early_stop_patience
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.early_stop_counter = 0

    def step(self, current_loss, model):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.early_stop_patience

    def load_best_model(self, model):
        model.load_state_dict(self.best_state_dict)

