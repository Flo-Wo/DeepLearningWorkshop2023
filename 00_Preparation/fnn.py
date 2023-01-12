# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms


class FNN(nn.Module):
    def __init__(self):
        pass


class DataLoaderFNN:
    def __init__(self, train_val_ratio: float, batch_size: int):
        """Constructor for the dataloader. The first call of this class
        will download the data automatically.

        Parameters
        ----------
        train_val_ratio : float
            Float between (0,1) how we will split our train data set
            into train and validation data. E.g. train_val_ratio = 0.8 means
            we use 80% for training and 20% for validation.
        batch_size : int
            Batch size, input tensors will have the shape (batch_size, input_shape).
        """
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        total_train_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            transform=data_transform,
            download=True,
        )
        len_train = int(len(total_train_dataset) * train_val_ratio)
        len_val = len(total_train_dataset) - len_train

        # split the original train data again into train + validation
        train_subset, val_subset = torch.utils.data.random_split(
            total_train_dataset,
            [
                len_train,
                len_val,
            ],
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=data_transform,
            download=True,
        )
        # define the DataLoader, which are going to be used in the stages of the NetworkTrainer
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_subset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_subset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

    def __call__(self, mode: str, **kwargs):
        """Return step specific DataLoader.

        Parameters
        ----------
        mode : str
            Possible are ``"train"``, ``"val"`` and ``"test"`` returning
            train, validation and test data respectively.

        Returns
        -------
        torch.utils.data.DataLoader
            Torch dataloader to sample batches of data.

        Raises
        ------
        NotImplementedError
            When called with an invalid argument.
        """
        # mode can be: train, val, test --> on this object we can use the call __iter__
        if mode == "train":
            return self.train_loader
        elif mode == "val":
            return self.val_loader
        elif mode == "test":
            return self.test_loader
        else:
            raise NotImplementedError(
                "Mode should be either 'train'/'val'/'test', check your call please."
            )

    def show_sizes(self):
        """Show the sizes of all datasets."""
        print("Train data: {}".format(len(self.train_loader.dataset)))
        print("Valid data: {}".format(len(self.val_loader.dataset)))
        print("Test data: {}".format(len(self.test_loader.dataset)))
