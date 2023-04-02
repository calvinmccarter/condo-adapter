import numpy as np
import torch

from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.loss_min = np.Inf
        self.state_dict = None

    def __call__(self, loss, model, epoch):
        if loss < self.loss_min:
            self.loss_min = loss
            self.epoch_min = epoch
            self.state_dict = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
