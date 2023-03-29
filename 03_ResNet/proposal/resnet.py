# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


class ResNetCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Implementation of a single ResNet cell according
        to He et al. (2016).

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        # TODO: 2,a)
        super(ResNetCell, self).__init__()
        # TODO: 2,a) Implement step 1
        # =================
        # we use full pre-activation
        self.step1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # stride of 2 is given by the original paper
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
        )
        # =================

        # TODO: 2,a) Implement Step 2
        # =================
        # we use the same size (out_channels, out_channels),
        # as in the (original) ResNet paper, again we use pre-activation
        self.step2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
        )
        # =================

        # TODO: 2,a) Check the projection case
        # =================
        # check whether we need a downsampling operation (implement by using
        # a conv1d operation)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # as we use a 1x1 kernel, padding is useless
                # we need to use twice the stride for the last operation
                # bias = False based on the pytorch internal version
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=4,
                    bias=False,
                )
            )
        else:
            self.shortcut = nn.Sequential()
        # =================

    def forward(self, input: torch.tensor):
        """Perform a forward propagation step on one input tensors.

        Parameters
        ----------
        x : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).

        Returns
        -------
        torch.tensor
            Extracted features.
        """
        # TODO: 2,a) Perform a forward pass with the steps implement above
        # =================
        shortcut = self.shortcut(input)
        res_inner = self.step1(input)
        residual = self.step2(res_inner)
        # =================
        return residual + shortcut


class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        """ResNet accoring to He et al. (2016).

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_classes : int
            Number of classes, i.e. output dimension.
        """
        super(ResNet, self).__init__()
        # TODO: 2,b) Implement the convoluton part
        # =================
        # first convolutional layer with a max-pooling layer (originally with 64
        # kernels)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # =================

        # TODO: 2,b) Add three ResNetCells
        # =================
        # define the residual layers by using our ResNetCell-class
        self.res1 = ResNetCell(8, 16)
        self.res2 = ResNetCell(16, 32)
        self.res3 = ResNetCell(32, 64)
        # =================

        # TODO: 2,b) Add a global average pooling and a flattening layer
        # =================
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        # =================

        # TODO: 2,b) Add a fully connected layer (feature compression)
        # =================
        self.fcl = nn.Sequential(nn.Linear(64, num_classes))
        # =================

    def forward(self, input: torch.tensor):
        """Perform a forward propagation step on one input tensors.

        Parameters
        ----------
        x : torch.tensor
            Batch of train/validation/test data with the shape
            (batch_size, input_shape).

        Returns
        -------
        torch.tensor
            Extracted features.
        """
        # TODO: 2,b) Use the layers implement above to define a forward pass
        # =================
        feat = self.conv_layer(input)
        res1 = self.res1(feat)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        gap = self.gap(res3)
        # =================
        return self.fcl(gap)


class ResNetPretrained(models.ResNet):
    def __new__(self, num_classes: int, feature_extract: bool = True, **kwargs):
        # TODO: 3,a) Load the network with pretrained weights
        # =================
        # use default weights for the newest trained netowork
        model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # =================

        # TODO: 3,b) Change the last layer of the pretrained network
        # =================
        # if we want to use the model only for feature extraction, we turn off
        # the gradients for all layers, except for the last one
        _set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, num_classes)
        return model_ft


def _set_parameter_requires_grad(model: torch.nn.Module, feature_extracting: bool):
    """Set the ``requires_grad`` flag for all layers of a given model.

    Parameters
    ----------
    model : torch.nn.Module
        (Pretrained) model.
    feature_extracting : bool
        Flag to decide whether we want to use the network for feature
        extraction only (i.e. turn off the gradients). If the flag is true,
        we turn off the gradients.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ResNetFactory:
    def __new__(self, pretrained: bool = False, **kwargs):
        if pretrained:
            return ResNetPretrained(**kwargs)
        return ResNet(**kwargs)


class DataLoaderResNet:
    def __init__(
        self,
        train_val_ratio: float,
        batch_size: int,
        data_dir: str = "./data",
    ):
        """Data loader providing images of ants or bees.

        Parameters
        ----------
        train_val_ratio : float
            Ratio to divide the training data into training
            and validation.
        batch_size : int
            Batch size for training and validation.
        data_dir : str, optional
            Folder containing the images, by default "./data".
        """
        # TODO: 1,b) & c) & d) Perform data augmentation on the training and validation data,
        # ensure the shape is in the end 224, normalize all samples
        # =================
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # mean, std for each of the three channels, see the pretrained version
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # =================

        # TODO: 1,c) and d) Resize the testing data to 224 and normalize all samples
        # =================
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # mean, std for each of the three channels, see the pretrained version
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # =================

        # TODO: 1,a) Load the data using the ImageFolder-class
        # =================
        # Image Folder uses the folder names to generate the labels automatically
        total_train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform_train
        )

        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform_test
        )
        # =================

        # TODO: 1,a) Split the training data into training and validation data
        # =================
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
        # =================

        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=False
        )
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        # NOTE: the classnames are encoded in
        # test_subset.classes
        # if you iterate through a dataset, you return
        # image, classes = next(iter(self.test_dataloader))

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


if __name__ == "__main__":
    test_input = torch.zeros(1, 3, 224, 224)
    my_net = ResNet(3, 2)
    test_output = my_net(test_input)
    print(test_output)
