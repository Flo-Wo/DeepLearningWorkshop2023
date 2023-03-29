# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
from resnet import DataLoaderResNet, ResNetFactory
from torch.utils.data import DataLoader
from utils import NetworkTrainer, NetworkUser, plot_performance
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def extract_conv_layers(model: nn.Module):
    # we will save the weights of the conv layers in this list
    model_weights = []
    # save the conv layers in this list
    conv_layers = []

    # save all the children of the model as a list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # TODO: Bonus b,i)
    # =================
    # Iterate through all the childen append all the conv layers and
    # their respective weights to the list
    # Attention: if a layer is a sequential layer, we need an additional iteration
    for child in model_children:
        if type(child) == nn.Conv2d:
            counter += 1
            model_weights.append(child.weight)
            conv_layers.append(child)
        elif type(child) == nn.Sequential:
            for sequ_child in child:
                for channel in sequ_child.children():
                    if type(channel) == nn.Conv2d:
                        counter += 1
                        model_weights.append(channel.weight)
                        conv_layers.append(channel)
    print("#Conv layer: {}".format(counter))
    # =================
    return conv_layers, model_weights


def vis_conv_kernel(model_weights: list) -> None:
    # TODO: Bonus b,ii)
    # =================
    # Create a figure and visualize the kernel of the *first* layer, the first layer has
    # 7x7 filter and a total of 64 channels
    plt.figure(figsize=(12, 12))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(
            8, 8, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap="gray")
        plt.axis("off")
        plt.savefig("./bonus_vis/kernels/filter.png")
    plt.title("Visualization of the convolutional kernels (weights of the first layer)")
    plt.show()
    # =================
    pass


def forward_image(conv_layers: list, img: torch.tensor):
    # TODO: Bonus b,iii)
    # Pass the image through all the layers, we start with the first
    # layer manually, as we need to pass the output of the previous layer
    # into the next current layer
    # =================
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # =================
    return results


def vis_feature_maps(outputs: list) -> None:
    # TODO: Bonus b,iv)
    # =================
    # Visualize only the first 8*8=64 channels from each layer, i.e.
    # create a 8x8 grid with a small image for each of the feature maps
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(20, 20))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 channels from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap="gray")
            plt.axis("off")
        print("Saving feature maps of layer {}".format(num_layer))
        plt.savefig("./bonus_vis/feature_maps/layer_{}.png".format(num_layer))
        plt.close()
    # =================
    pass


def bonus_visualization(model: nn.Module, data_loader: DataLoader):
    # extract the convolutional layers of the model
    conv_layers, model_weights = extract_conv_layers(model)

    for weight, conv in zip(model_weights, conv_layers):
        print("layer {}, shape: {}".format(conv, weight.shape))

    # visualize the convolutional kernel of the first layer
    vis_conv_kernel(model_weights)

    img_batch, _ = next(iter(data_loader("train")))
    # load the first image of the batch, we need to
    img = img_batch[0, :, :, :]
    # img has the shape (3, 244, 244)
    # --> the color channel has to be the last one
    img_to_show = img.permute(2, 1, 0)
    plt.imshow(img_to_show)
    plt.title("Input Image (color might be strange, due to normalization)")
    plt.show()

    # unsqueeze the image, to get a batch of size 1
    outputs = forward_image(conv_layers, img_batch[0, :, :, :].unsqueeze(0))

    # visualize 64 features from each layer (deeper layers have more features which
    # will be ignored)
    vis_feature_maps(outputs)


if __name__ == "__main__":
    # TODO: 2,c) Instantiate your custom data loader and train your network
    # =================
    data_loader = DataLoaderResNet(0.9, 64)
    custom_trainer = NetworkTrainer(
        ResNetFactory,
        nn.CrossEntropyLoss,
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        data_loader,
    )
    network_params_custom = {"in_channels": 3, "num_classes": 2}

    # TODO: Bonus, a) Sum correct preditions to obtain the accuracy in percentage points
    def sum_correct_preds(
        prediction_output: torch.tensor,
        target_output: torch.tensor,
    ):
        # could also use max, returns: value, idx = torch.max()
        preds = torch.argmax(prediction_output, dim=1)
        return torch.sum(preds == target_output.data)

    path_custom = custom_trainer.train_network(
        10,
        network_params_custom,
        loss_params={},
        optimizer_params={},
        scheduler_params={},
        sum_correct_preds=sum_correct_preds,
    )
    # =================

    # TODO: 2,d) Evaluate the network on the validation dataset
    # =================
    user = NetworkUser(ResNetFactory, nn.CrossEntropyLoss, data_loader)
    cache_custom = user.test_network(
        network_params_custom,
        {},
        path_to_network=path_custom,
        plot_loss=False,
        sum_correct_preds=sum_correct_preds,
    )
    # =================

    # TODO: 3,c) Train the last layer of the pretrained network
    # =================
    network_params_pretrained = {
        "pretrained": True,
        "extract_features": True,
        "in_channels": 3,
        "num_classes": 2,
    }

    path_pretrained = custom_trainer.train_network(
        10,
        network_params_pretrained,
        loss_params={},
        optimizer_params={},
        scheduler_params={},
        sum_correct_preds=sum_correct_preds,
    )
    # =================

    # TODO: 3,d) Evaluate the pretrained network on the validation data
    # =================
    cache_pretrained = user.test_network(
        network_params_pretrained,
        {},
        path_to_network=path_pretrained,
        plot_loss=False,
        sum_correct_preds=sum_correct_preds,
    )
    # =================

    # TODO: 2,d) and 3,d) Plot the results for both of the networks
    # =================
    plot_performance(cache_custom, cache_pretrained)
    # =================

    # TODO: Bonus b,v)
    # =================
    # bonus_visualization(ResNetFactory(True, num_classes=2), data_loader)
    # =================
