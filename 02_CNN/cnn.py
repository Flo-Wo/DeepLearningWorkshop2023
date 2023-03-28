# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import numpy as np


# TODO: Bonus (cf. first assignment), we might want to change the default's
# behavior to init the weights and biases
def init_weights(self, layer):
    # ===============
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.Conv2d):
        pass
    # ===============


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        # TODO: 2,a) Build the CNN part (we use shared weights, so you only need
        # to initialize one of the two parts)
        # We implement the feature extraction part
        # ===============
        # Build the CNN part
        self.features = nn.Sequential(
            # first cnn layer
            nn.Conv2d(3, 16, 10),
            # using inplace=True can reduce the internal memory footprint
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # second cnn layer
            nn.Conv2d(16, 32, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # third cnn layer
            nn.Conv2d(32, 32, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # ===============

        # TODO: 2,b) Build the Compression part by using a fully connected layer
        # ===============
        # Define the fully connected layer to compress the features maps
        self.compression = nn.Sequential(nn.Linear(2048, 800), nn.ReLU())
        # ===============

        # TODO. 2,c) Add the similarity measure part of our SNN
        # ===============
        # define the last layer via fcl + sigmoid --> similarity measure
        self.similarity = nn.Sequential(nn.Linear(800, 1), nn.Sigmoid())
        # ===============

        # self.apply(init_weights)

    def _forward_once(self, x: torch.tensor):
        """Perform a forward propagation step on one input tensors.
        Implements the shared weights part.

        Parameters
        ----------
        x : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).

        Returns
        -------
        torch.tensor
            Extracted features, shape depends on the design of the CNN-part.
        """
        # TODO: 2,a) Call the feature feature extraction and the
        # compression part (hint: read the docstring)

        # ===============
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        print(x.shape)
        # ===============
        return self.compression(x)

    def forward(self, input1: torch.tensor, input2: torch.tensor):
        """Perform a forward propagation step on the two input tensors.

        Parameters
        ----------
        input1 : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).
        input2 : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).

        Returns
        -------
        torch.tensor
            Predicted output with the shape ``(batch_size, ?)``. For the shape
            take a look at the assignment.
        """

        # TODO: 2, c) Use the _forward_once function appropriately and
        # call the similiarity part afterwards (hint: read the docstring)
        # ===============
        out1 = self._forward_once(input1)
        out2 = self._forward_once(input2)
        dist = torch.abs(out1 - out2)
        # ===============
        return self.similarity(dist)


class DataLoaderSNN:
    def __init__(
        self,
        train_meta_path: str,
        train_path: str,
        test_meta_path: str,
        test_path: str,
        train_val_ratio: float,
        batch_size: int,
        transform=None,
    ):
        """Generic DataLoader for the signature verification task.

        Parameters
        ----------
        train_meta_path : str
            Path the table with the filenames and the labels of the train data.
        train_path : str
            Path to the folder with the training images.
        test_meta_path : str
            Path the table with the filenames and the labels of the test data.
        test_path : str
            Path to the folder with the test images.
        train_val_ratio : float
            Float between (0,1) how we will split our train data set
            into train and validation data. E.g. train_val_ratio = 0.8 means
            we use 80% for training and 20% for validation.
        batch_size : int
            Batch size, input tensors will have the shape (batch_size, input_shape).
        transform : torchvision.transform, optional
            Optional transformer or multiple transformers via ``transforms.Compose()``
            to optionally premodify the images (e.g. reduce the resolution),
            by default None.
        """
        # TODO: 1,b) Check the function docstring and take a look at the DataLoader
        # used in the first assignment if you need a hint
        # First we load the dataset, then split the training set into training and
        # testing by performin a random split and afterwards we use the DataLoader
        # class to construct three dataloaders ("train", "test", "val")
        # ===============
        total_train_dataset = SignatureDataset(
            train_meta_path, train_path, transform=transform
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
        test_dataset = SignatureDataset(test_meta_path, test_path, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_subset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_subset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        # ===============

    def __call__(self, mode: str):
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


class SignatureDataset(Dataset):
    def __init__(self, annotations_path: str, data_path: str, transform=None):
        """Generic Dataset of the signature verification task.
        Used to load the train and test data.

        Parameters
        ----------
        annotations_path : str
            Path to the table containing the filenames and the labels.
        data_path : str
            Path to the folder containing the images.
        transform : torchvision.transform, optional
            Optional transformer or multiple transformers via ``transforms.Compose()``
            to optionally premodify the images (e.g. reduce the resolution),
            by default None.
        """
        # TODO: 1,a) Check the docstring of the function
        # Load the csv-table, compute the length of the dataset and save the transformers
        # as a class attribute
        # ===============
        # NOTE: give the self.____ part inside the solution (columns fully given)
        self.annotations = pd.read_csv(annotations_path)
        self.annotations.columns = ["reference", "questioned", "label"]
        self.data_path = data_path
        self.len_dataset = len(self.annotations)
        self.transform = transform
        # ===============

    def __getitem__(self, index: int):
        """Our Dataset should work like a generator.
        Implementing ``__getitem__`` allows as to call ``enumerate()``
        or ``next()`` on our dataset. Attention, you should **always**
        implement the ``__len__`` attribute.

        Parameters
        ----------
        index : int
            Index of the sample, this is internally set by pytorch
            automatically. The index represents a row ID of our metadata
            table.

        Returns
        -------
        torch.tensor, torch.tensor, torch.tensor
            Three tensors: reference image, questioned image and the
            corresponding label.
        """
        # TODO: 1,a) Check the docstring of the function for a more detailed
        # description
        # We read the two images using the function below,
        # load the label (we transform the label to a float
        # and afterwards to a torch tensor), the optional transformers are applied
        # and the two tensors and their corresponing label is returned

        # ===============
        im1 = self._read_image(index, 0)
        im2 = self._read_image(index, 1)
        y_label = torch.from_numpy(
            np.array([self.annotations.iat[index, 2]], dtype=np.float32)
        )
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        return im1, im2, y_label
        # ===============

    def _read_image(self, sample_index: int, im_index: int):
        return io.imread(
            os.path.join(self.data_path, self.annotations.iat[sample_index, im_index])
        )

    def __len__(self):
        return self.len_dataset


# TODO: Bonus
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        pass

    def forward(self, input1: torch.tensor, input2: torch.tensor, target: torch.tensor):
        pass


# for debbuging
if __name__ == "__main__":
    net = SNN()
    print(net)
    test_input = torch.ones((1, 3, 100, 100))
    _ = net(test_input, test_input)
    # print(list(net.parameters()))
