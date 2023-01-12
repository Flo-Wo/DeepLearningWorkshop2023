# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class CacheLoss:
    metadata: np.array
    test_loss: float
    label: str = "None"


def plot_performance(*model_caches: CacheLoss, save_to_path: str = None):
    """Performance plotter for a single model or multiple models.

    Parameters
    ----------
    model_caches : Union[CacheLoss, list[CacheLoss]]
        Cached data of a single model or of multiple models in form of
        a list.
    save_to_path : str
        Given the full path (folder + filanem) saves the plot as a .pdf-file.
    """
    title = "Test losses"
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(model_caches))))
    for idx, model in enumerate(model_caches):
        train_loss = model.metadata[:, 0]
        validation_loss = model.metadata[:, 1]
        color = next(colors)
        label = "({}) {}".format(idx, model.label)
        plt.plot(
            train_loss,
            label=label + ": train",
            color=color,
            marker="x",
        )
        plt.plot(validation_loss, label=label + ": val", color=color, marker="o")
        title += "\n{}: {:1.2e}".format(label, model.test_loss)
    plt.title(r"{}".format(title))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.gca().ticklabel_format(useMathText=True)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    if save_to_path:
        plt.savefig(save_to_path + ".pdf")
    plt.show()


# one can turn on active latex formatting, but makes the plotting process really slow
# from matplotlib import rc
# rc("text", usetex=True)
