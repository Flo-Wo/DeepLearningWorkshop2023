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
    def __init__(
        self,
        network_class: nn.Module,
        loss_function: nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ):
        """Decouples the train and test process. Load parameters and evaluate the model.

        Parameters
        ----------
        network_class : nn.Module
            Class of the network, e.g. ``DNN``.
        loss_function : nn.Module
            Loss function, e.g. ``torch.nn.CrossEntropyLoss``.
        data_loader : torch.utils.data.DataLoader
            Data loader to provide the test dataset via:
            ```python
            test_data = data_loader('test')
            ```
        """
        self.network_class = network_class
        self.loss_function = loss_function
        self.data_loader = data_loader
        self.path_to_network = None
        self.folder = "./metadata"

    def test_network(
        self,
        network_params: dict,
        loss_params: dict,
        path_to_network: str,
        plot_loss: bool = True,
        folder: str = "./metadata/",
        sum_correct_preds: callable = lambda pred_output, target_output: -1,
    ):
        """Evaluate the model on the test dataset.

        Parameters
        ----------
        network_params : dict
            Parameters of the network, for a detailed description see ``network_params``
           in the ``NetworkTrainer``-class.
        loss_params : dict
            Parameters used for the loss function.
        path_to_network : str
            String of the previously trained network, given by the ``NetworkTrainer``, e.g. by
            ```python
            >>> path = trainer.train_network(
            >>>     12,
            >>>     network_params,
            >>>     loss_params,
            >>>     {"lr": 0.001},
            >>>     scheduler_params=dict(factor=0.9, patience=1),
            >>>     )
            >>> user = NetworkUser(DNN, torch.nn.CrossEntropyLoss, data_loader)
            >>> user.test_network(network_params, loss_params, path)
            ```
            you can load the network automatically.
        plot_loss : bool
            Plot the train and validation loss curves, by default True.
        folder : str, optional
            Folder to store meta information, by default "./metadata/".
        sum_correct_preds : callable
            Function which is called after every batch iteration in the test
            cycle and is used to manually compute the accuracy of a model.
            The function always has the parameters ``prediction_output`` and
            ``target_output`` both beeing of the type ``torch.tensor``.
            The default is the const. function -1 and thus, without a callback the accuracy
            is always negative.
            Given a prediction and a target, sum_correct_preds computes the absolute
            number of correct predictions. Internally the sum of all those values is
            averaged and used to compute the accuracy in percentage points.
        Returns
        -------
        CacheLoss
            Helper class to save the test loss, metadata and the label. This will simplify
            the plotting procedure if we want to compare multiple networks.
        """
        self.path_to_network = path_to_network
        self.folder = folder

        self.metadata = self._load_metadata()

        model = self._load_network(network_params, self.path_to_network, self.folder)
        criterion = self.loss_function(**loss_params)

        test_dataset = self.data_loader("test")
        len_test_data = len(test_dataset.dataset)

        test_loss = 0
        test_correct = 0
        # turn off backpropagation for more efficiency
        with torch.no_grad():
            for batch_idx, (*input_data, target_output) in enumerate(
                tqdm(test_dataset, "Test")
            ):
                predicted_output = model(*input_data)
                loss = criterion(predicted_output, target_output)
                test_loss += loss.item()

                test_correct += sum_correct_preds(predicted_output, target_output)

        test_loss /= len_test_data
        print(
            "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                test_correct,
                len_test_data,
                100.0 * test_correct / len_test_data,
            )
        )
        cache_loss = CacheLoss(self.metadata, test_loss, label=str(self.network_class))
        if plot_loss:
            plot_performance(cache_loss)
        return cache_loss

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
