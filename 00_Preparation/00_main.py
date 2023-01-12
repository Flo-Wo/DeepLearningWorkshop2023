# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

from fnn import DataLoaderFNN
import matplotlib.pyplot as plt


def show_images(data_loader):
    # TODO: 1,a) Visualize the first 6 images of the given dataset
    # we load the first batch of images and labels by calling next()
    # and plot the images using the _plot_images function
    pass


def _plot_images(example_data, example_targets, predicted_targets=None):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i], cmap="gray", interpolation="none")
        title = "True: {}".format(example_targets[i])
        if predicted_targets is not None:
            title += "\nPrediction: {}".format(predicted_targets[i])
        plt.title(title)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def analyze_distribution(data_loader: DataLoaderFNN, relative: bool = False):
    # TODO: 1,b) We call the helper function _dist_dataset which sums the
    # occurency of all labels (by using torch.bincount() and computes the average
    # The resulting frequencies are visualized in a matplotlib bar chart
    pass


if __name__ == "__main__":
    data_loader = DataLoaderFNN(train_val_ratio=0.8, batch_size=100)
    data_loader.show_sizes()

    # TODO: 1,a)
    # Load 6 images from the train dataset, their corresponding labels
    # and visualize them in a plot
    # ============
    # YOUR CODE
    # ============

    # TODO: 1,b)
    # Show the distribution of labels for each dataset (train, test, validation)
    # Visualize the frequency each label has in a histogram
    # ============
    # YOUR CODE
    # ============

    # TODO: 1,c)
    # What is the size of the images?
