# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
from utils import NetworkTrainer, NetworkUser
from cnn import SNN, DataLoaderSNN
import torch
import torch.nn as nn
from torchvision import transforms

if __name__ == "__main__":
    # TODO: 2,d)
    # ===============
    # we need to resize the data to get a uniform shape and transform it into tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
        ]
    )
    # ===============

    # TODO: 2,d) Instantiate your DataLoader
    # ===============
    data_loader = DataLoaderSNN(
        train_meta_path="./data/train_data.csv",
        train_path="./data/train_data/",
        test_meta_path="./data/test_data.csv",
        test_path="./data/test_data/",
        train_val_ratio=0.85,
        batch_size=100,
        transform=transform,
    )
    # ===============

    # TODO: Instantiate a trainer with the BCE-loss, an optimizer and a scheduler (e.g.
    # torch.optim.lr_schuler.ReduceLRonPlateau)
    # ===============
    trainer = NetworkTrainer(
        SNN,
        torch.nn.BCELoss,
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        data_loader,
        folder="./metadata/",
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
        pass
        # ===============
        # use p=0.5 as a threshold, if value > p => return 1, else 0
        # ===============

    # TODO: 2,d) Train your network using the trainer instantiated above
    # ===============
    path = trainer.train_network(
        10,
        network_params,
        loss_params={},
        optimizer_params={},
        scheduler_params={},
        sum_correct_preds=sum_correct_preds,
    )
    # ===============

    # TODO. 2,e) Evaluate your network
    # ===============
    user = NetworkUser(SNN, torch.nn.BCELoss, data_loader)
    _ = user.test_network(
        {},
        {},
        path_to_network=path,
        plot_loss=True,
        sum_correct_preds=sum_correct_preds,
    )
    # ===============
