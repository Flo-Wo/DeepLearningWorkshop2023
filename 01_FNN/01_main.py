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
        # ============

        # ============

        title = "True: {}".format(example_targets[i])
        if predicted_targets is not None:
            title += "\nPrediction: {}".format(predicted_targets[i])
        plt.title(title)

        plt.axis("off")
    fig.savefig("./01_sheet/imgs/numbers.pdf")
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
    # ============
    trainer = NetworkTrainer(
        FNN,
        # TODO,
        # TODO,
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
        "input_shape": # TODO,
        "hidden_shape": # TODO,
        "num_classes": # TODO,
        "activation_func": # TODO,
        # TODO: Bonus,b)
        # "use_batch_norm": True,
    }
    loss_params = {}
    # ============
    # TODO: Bonus, a)
    # ============

    # ============

    # TODO: Bonus, b) & d) 
    # ============

    # ============

    # TODO: 4,b)
    # ============
    path = trainer.train_network(
        12,
        network_params,
        loss_params,
        {
            "lr": # TODO,
        },
        scheduler_params=dict(factor=0.9, patience=1),
        # TODO: Bonus,a)
        # sum_correct_preds=sum_correct_preds,
        # TODO: Bonus,d)
        # regularization=regularization("L2"),
    )
    print(path)
    # ============

    # compute and return loss on the test data --> use NetworkUser class
    # TODO: 4,c) Uncomment the following code snippet
    # ============
    # user = NetworkUser(FNN, torch.nn.CrossEntropyLoss, data_loader)
    # cache_default = user.test_network(
    #     network_params, 
    #     loss_params, 
    #     path, 
    #     plot_loss=False, 
    #     # TODO: Bonus,a)
    #     # sum_correct_preds=sum_correct_preds,
    # )
    # plot_performance(cache_default)
    # ============

    # TODO: 4,d) Check the description of the NetworkUser's ``get_predictor``-function
    # We extract the prediction function of the Networkuser to manually test our network
    # on a few images
    # ============

    # ============

    # TODO: 4,d)
    # We need to reshape our tensor back to images by using .reshape((28,28))
    # ============
    # examples = iter(data_loader("test"))
    # example_data, example_targets = next(examples)

    # ============
