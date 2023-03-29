# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

from fnn import FNN, DataLoaderFNN
from utils import NetworkTrainer, NetworkUser
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.plotting import plot_performance

# for MNIST we have
# mnist_mean, mnist_std = 0.1307, 0.3081


def show_images(data_loader: DataLoaderFNN):
    # TODO: 1,a) Visualize the first 6 images of the given dataset
    # we load the first batch of images and labels by calling next()
    # and plot the images using the _plot_images-function
    examples = iter(data_loader)
    example_data, example_targets = next(examples)
    _plot_images(example_data, example_targets)


def _plot_images(
    example_data: torch.Tensor,
    example_targets: torch.Tensor,
    predicted_targets: torch.Tensor = None,
):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        # plt.imshow(example_data[i], cmap="gray", interpolation="none")
        # TODO: 4,d)
        # Reshape the tensors to images again
        plt.imshow(example_data[i].reshape((28, 28)), cmap="gray", interpolation="none")

        title = "True: {}".format(example_targets[i])
        if predicted_targets is not None:
            title += "\nPrediction: {}".format(predicted_targets[i])
        plt.title(title)

        plt.axis("off")
    # fig.savefig("./01_sheet/imgs/numbers.pdf")
    plt.show()


def analyze_distribution(data_loader: DataLoaderFNN, relative: bool = False):
    # TODO: 1,b) We call the helper function _dist_dataset which sums the
    # occurency of all labels (by using torch.bincount() and computes the average
    # The resulting frequencies are visualized in a matplotlib bar chart
    num_train = _dist_dataset(data_loader("train"), relative)
    num_val = _dist_dataset(data_loader("val"), relative)
    num_test = _dist_dataset(data_loader("test"), relative)

    labels = torch.arange(0, 10)

    plt.bar(labels, num_train, label="training")
    plt.bar(labels, num_val, label="validation")
    plt.bar(labels, num_test, label="testing")
    plt.legend(loc="upper right")
    plt.xlabel("labels")
    plt.ylabel("number of samples")
    plt.show()


def _dist_dataset(dataset: DataLoaderFNN, relative: bool = False):
    num = torch.zeros(10)
    for _, batch_label in dataset:
        num += torch.bincount(batch_label)
    if not relative:
        return num
    return num / torch.sum(num)


if __name__ == "__main__":
    data_loader = DataLoaderFNN(train_val_ratio=0.8, batch_size=100)
    data_loader.show_sizes()

    # TODO: 1,a)
    # Load 6 images from the train dataset, their corresponding labels
    # and visualize them in a plot
    # ============
    show_images(data_loader("train"))
    # ============

    # TODO: 1,b)
    # Show the distribution of labels for each dataset (train, test, validation)
    # Visualize the frequency each label has in a histogram
    # ============
    analyze_distribution(data_loader)
    # ============

    # TODO: 1,c)
    # What is the size of the images
    # Answer: the images have the size 28x28

    # TODO: 4,a) Pass the Loss funtion into the NetworkTrainer
    # TODO: 4,b) Pass the optimizer into the NetworkTrainer
    # We use the CrossEntropyLoss (=softmax + NLL-loss)
    # and thus don't need an activation function in our last layer
    # ============
    trainer = NetworkTrainer(
        FNN,
        torch.nn.CrossEntropyLoss,
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        data_loader,
        folder="./metadata/",
    )
    # ============

    # TODO: 4,b)
    # We pass all arguments the constructor takes into the trainer by using
    # a dictionary
    # ============
    network_params = {
        "input_shape": 28 * 28,
        "hidden_shape": 500,
        "num_classes": 10,
        "activation_func": nn.ReLU,
        # TODO: Bonus,b)
        # "use_batch_norm": True,
    }

    loss_params = {}
    # ============

    # TODO: Bonus, a)
    # We extract the label by taking the index with the maximum probability
    # Check for equality and sum the number of correct predictions
    # ============
    def sum_correct_preds(pred_output: torch.tensor, target_output: torch.tensor):
        pred = pred_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        return pred.eq(target_output.view_as(pred)).sum().item()

    # ============

    # TODO: Bonus, d)
    # Add a custom regularization function to your network
    # ============
    def regularization(type="L1", damping: float = 0.001):
        if type == "L1":

            def add_to_loss(model):
                return damping * sum(p.abs().sum() for p in model.parameters())

        elif type == "L2":

            def add_to_loss(model):
                return damping * sum(p.pow(2.0).sum() for p in model.parameters())

        else:
            raise NotImplementedError("Regularization type is not implemented")
        return add_to_loss

    # ============

    # TODO: 4,b)
    # ============
    path = trainer.train_network(
        12,
        network_params,
        loss_params,
        {"lr": 0.001},
        # TODO: Bonus,d)
        # {"lr": 0.001, "weight_decay": 0.001}
        scheduler_params=dict(factor=0.9, patience=1),
        # TODO: Bonus,a)
        sum_correct_preds=sum_correct_preds,
        # TODO: Bonus,d)
        # regularization=regularization("L2"),
    )
    print(path)
    # path_default = "input_shape_784_hidden_shape_500_num_classes_10_activation_func_<class 'torch.nn.modules.activation.ReLU'>_<class 'fnn.FNN'>"
    # ============

    # compute and return loss on the test data --> use NetworkUser class
    # path = "input_shape_784_hidden_shape_500_num_classes_10_activation_func_<class 'torch.nn.modules.activation.ReLU'>_<class 'dnn.DNN'>"
    # TODO: 4,c) Uncomment the following code snippet
    # ============
    user = NetworkUser(FNN, torch.nn.CrossEntropyLoss, data_loader)
    cache_default = user.test_network(
        network_params,
        loss_params,
        path,
        plot_loss=False,
        # TODO: Bonus, a)
        sum_correct_preds=sum_correct_preds,
    )
    plot_performance(cache_default)
    # ============

    path2 = "input_shape_784_hidden_shape_500_num_classes_10_activation_func_<class 'torch.nn.modules.activation.ReLU'>_use_batch_norm_True_<class 'dnn.DNN'>"
    cache_batch_norm = user.test_network(
        dict({"use_batch_norm": True}, **network_params),
        loss_params,
        path2,
        plot_loss=False,
        sum_correct_preds=sum_correct_preds,
    )
    plot_performance([cache_default, cache_batch_norm])

    # TODO: 4,d) Check the description of the NetworkUser's ``get_predictor``-function
    # We extract the prediction function of the Networkuser to manually test our network
    # on a few images
    # ============
    predict = user.get_predictor(network_params, path)
    # ============

    # TODO: 4,d)
    # We need to reshape our tensor back to images by using .reshape((28,28))
    # ============
    examples = enumerate(data_loader("test"))
    _, (example_data, example_targets) = next(examples)
    # we optimized our code by combining softmax and neg-log-likelihood (nll)-loss
    # need softmax again + extract max to get the predicted label
    softmax = nn.Softmax(dim=1)
    pred_labels = torch.argmax(softmax(predict(example_data)), dim=1)
    # print(pred_labels)
    _plot_images(example_data, example_targets, pred_labels)
    # ============
