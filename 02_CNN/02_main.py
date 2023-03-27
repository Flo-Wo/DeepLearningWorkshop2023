# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
from utils import NetworkTrainer, NetworkUser
from cnn import SNN, DataLoaderSNN
import torch
import torch.nn as nn
from torchvision import transforms
from utils.plotting import plot_performance

if __name__ == "__main__":
    # TODO: 2,d)
    # ===============

    # ===============

    # TODO: 2,d) Instantiate your DataLoader
    # ===============
    data_loader = DataLoaderSNN(
        # TODO
    )
    # ===============

    # TODO: Instantiate a trainer with the BCE-loss, an optimizer and a scheduler (e.g.
    # torch.optim.lr_schuler.ReduceLRonPlateau)
    # ===============
    trainer = NetworkTrainer(
        # TODO
    )
    network_params = {}
    # ===============

    # TODO: Bonus,a) Compute the accuracy in percentage points
    # ===============
    def sum_correct_preds(
        prediction_output: torch.tensor,
        target_output: torch.tensor,
        threshold: float = 0.5,
    ):
        """Given a prediction and a target, compute the absolute number of correct predictions.
        Helper function to internally compute the accuracy in percentage points (this function
        is called batch-wise and the sum of the results will be averaged after each epoch).

        Parameters
        ----------
        prediction_output : torch.tensor
            Prediction of the network.
        target_output : torch.tensor
            Target the network should fit.
        threshold : float, optional
            Decision threshold, by default 0.5. If the prediction value is > threshold,
            we return 1, otherwise we return 0. By varying the value of the threshold,
            we can enforce how certain the network has to be.

        Returns
        -------
        int
            Number of correct predictions (depends on the decision threshold).
        """
        # ===============
        # ===============
        pass

    # TODO: 2,d) Train your network using the trainer instantiated above
    # ===============
    path = trainer.train_network(
        # TODO
    )
    # ===============

    # TODO. 2,e) Evaluate your network
    # ===============
    user = 
    _ = user.test_network(
        # TODO
    )
    # ===============
