# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms


def _dense_layer(
    input_shape: int,
    output_shape: int,
    activation_func: torch.nn = None,
    batch_normalize: bool = False,
):
    """Create a generic dense layer, which consists of:
        - linear layer with the shape (input_shape, output_shape)
       (- batch normalization layer, see BONUS)
        - activation function (optional)

    Parameters
    ----------
    input_shape : int
        Size of the input vectors.
    output_shape : int
        Size the output vector should have.
    activation_func : torch.nn, optional
        Activation function used after the linear layer, by default None
        as we might want to define layers without an activation function.
    batch_normalize : bool, optional
        Flag to additionally add a batch normalization layer (see Bonus
        tasks), by default False.

    Returns
    -------
    nn.Sequential
        Sequential containing the modules listed above.
    """
    # TODO: 2,a) Take a look at the function header
    # Attention: the flag batch_normalize is part of the bonus task, so you
    # can ignore this parameter
    steps = []
    # ============

    # ============
    return nn.Sequential(*steps)


def _init_weights(layer: torch.nn):
    """[BONUS] Generically initialize the weights of a layer based on its type.

    Can be used inside the constructor via
    ```python
    self.apply(_init_weights)
    ```
    **after** you defined all layers.

    Parameters
    ----------
    layer : torch.nn
        Generic layer of our network.
    """
    # TODO: Bonus, c) We call the function self.apply(_init_weights)
    # inside our constructor and all the layers are passed into the function
    # We check for the type of the layer and change it's default weights and biases

    # print(type(layer))
    # ============

    # ============


class FNN(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_shape: int,
        num_classes: int,
        activation_func,
        use_batch_norm: bool = False,
        **kwargs,
    ):
        """Constructor method for the FNN.

        Parameters
        ----------
        input_shape : int
            Shape of the input tensors that we will pass into the network.
        hidden_shape : int
            Number of neurons in the hidden layer.
        num_classes : int
            Number of outputs, in our case the number of classes we want
            to predict.
        activation_func : class
            Class name of the activation function.
            Attention, do *not* directly pass a callable into the
            constructor, e.g. use torch.nn.ReLU.
        use_batch_norm : bool, optional
            Flag to turn on batch normalization layer (see Bonus task),
            by default False.
        """
        super(FNN, self).__init__()

        # TODO: 2,b) Define two hidden layers by calling your _dense_layer method,
        # add an additional output layer
        # ============

        # ============

        # TODO: Bonus,c) Change the default initialization for the network's
        # weights and biases
        # ============
        # self.apply(_init_weights)
        # ============

    def forward(self, input: torch.tensor):
        """Perform a forward propagation step on the input tensor.

        Parameters
        ----------
        input : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).

        Returns
        -------
        torch.tensor
            Predicted output with the shape ``(batch_size, num_class)``.
        """
        # TODO: 2,c) Take a look at the function header, implement
        # the forward pass
        # ============
        pass
        # ============


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
                # TODO: 2,d) Uncomment the following two lines
                # transform the images to a torch tensor and afterwards
                # to vector shaped tensor
                # ============
                # transforms.ToTensor(),
                # transforms.Lambda(lambda x: x.reshape(28 * 28)),
                # ============
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
