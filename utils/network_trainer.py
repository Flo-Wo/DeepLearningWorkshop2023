# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.decorators import log_time


class NetworkTrainer:
    def __init__(self):
        pass

    # decorator to stop the training time
    @log_time
    def train_network(self):
        pass

    def _save(self, state_dict: dict, network_params: dict):
        """Save model params and metadata."""
        filename = self._generate_filename(self.network_class, network_params)
        full_path = self.folder + filename
        torch.save(state_dict, full_path + "_model.pt")
        np.save(full_path + "_loss.npy", self.metadata)
        return filename

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Helper function to extend the metadata."""
        self.metadata[epoch, :] = np.array([train_loss, val_loss])

    def _generate_filename(self, network_class: nn.Module, network_params: dict):
        """Helper function to generate a unique filename based on the network's parameters"""
        filename = ""
        for key, value in network_params.items():
            filename += "{}_{}_".format(key, value)
        return filename + str(network_class)
