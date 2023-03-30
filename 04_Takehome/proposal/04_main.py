# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
from utils import NetworkTrainer, NetworkUser
from rnn import SentimentRNN, SentimentLSTM, DatasetIMDB
from nlp_utils import prepare_data

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


def sum_correct_preds(
    prediction_output: torch.tensor,
    target_output: torch.tensor,
):
    """
    Given a prediction and a target, compute the absolute number of correct predictions.
    Helper function to internally compute the accuracy in percentage points (this function
    is called batch-wise and the sum of the results will be averaged after each epoch).
    """
    # TODO: BONUS,a)
    # We need sigmoid here, as we included sigmoid in our loss function
    # =================
    rounded_preds = torch.round(torch.sigmoid(prediction_output))
    correct = (
        rounded_preds == target_output
    ).float()  # convert into float for division
    return correct.sum()
    # =================


if __name__ == "__main__":
    # TODO: 2,c)
    # =================
    data_loader = DatasetIMDB(
        train_to_val_ratio=0.8,
        train_val_to_test_ratio=0.5,
        batch_size=64,
    )

    custom_trainer = NetworkTrainer(
        # SentimentRNN,
        SentimentLSTM,
        nn.BCEWithLogitsLoss,  # we combine sigmoid with the BCE loss
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        data_loader,
    )

    network_params_custom = {
        "input_dim": len(data_loader.vocabulary) + 1,  # +1 for padding
        "embedding_dim": 64,
        "hidden_dim": 256,
        "output_dim": 1,
    }

    path_custom = custom_trainer.train_network(
        10,
        network_params_custom,
        loss_params={},
        optimizer_params={},
        scheduler_params={},
        sum_correct_preds=sum_correct_preds,
    )

    # =================

    # TODO: 2,d)
    # =================
    user = NetworkUser(SentimentLSTM, nn.BCEWithLogitsLoss, data_loader)
    cache_custom = user.test_network(
        network_params_custom,
        {},
        path_to_network=path_custom,
        plot_loss=False,
        sum_correct_preds=sum_correct_preds,
    )
    # =================

    # TODO: BONUS,b)
    # =================
    sentiment_analysis = user.get_predictor(network_params_custom, path_custom)
    feedback_raw = [
        # TODO
    ]
    feedback_processes, _, _ = prepare_data(feedback_raw, [1])
    feedback_tensor = torch.from_numpy(feedback_processes)
    feedback_predict = sentiment_analysis(feedback_tensor)
    print(feedback_predict)
    # =================
