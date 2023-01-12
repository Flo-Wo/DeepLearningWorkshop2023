# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils.plotting import CacheLoss, plot_performance


class NetworkUser:
    def __init__(self):
        pass

    def test_network(self):
        pass

    def get_predictor(
        self,
        network_params: dict,
        path_to_network: str,
        folder: str = "./metadata/",
    ):
        """Load network and return a function that can be
        used to make predictions:

        ```python
        >>> predict = trainer.get_predictor(network_params)
        >>> predicted_labels = predict(my_input)
        ```

        Parameters
        ----------
        network_params : dict
            Parameters of the network.
        path_to_network : str
            String of the previously trained network, given by the ``NetworkTrainer``.
        folder : str, optional
            Folder to store meta information, by default "./metadata/".
        Returns
        -------
        Callable
            Function to evaluate the network and make predictions, attention
            we have a ``torch.no_grad()`` as default.
        """
        model = self._load_network(network_params, path_to_network, folder)

        def _predict(*input):
            with torch.no_grad():
                return model(*input)

        return _predict

    def _load_metadata(self):
        return np.load(self.folder + self.path_to_network + "_loss.npy")

    def _load_network(self, network_params: dict, path_to_network: str, folder: str):
        """Helper function to internally load the network."""
        model = self.network_class(**network_params)
        # NOTE: load_state_dict takes a dict, NOT a path
        # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        model.load_state_dict(torch.load(folder + path_to_network + "_model.pt"))
        model.eval()
        return model
