# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from nlp_utils import prepare_data
import logging


class SentimentRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_rnn_layers: int = 1,
    ):
        """RNN for sentiment analysis.

        The networks consists of:
            - Embedding Layer
            - RNN layer(s)
            - Linear layer

            Parameters
            ----------
            input_dim : int
                Input dimension of the Embebdding layer.
            embedding_dim : int
                Size of the embedding vector.
            hidden_dim : int
                The number of features in the hidden state h.
            output_dim : int
                Dimension of the output, i.e. the result of the
                linear layer.
            num_rnn_layers : int, optional
                Number of recurrent layers, by default 1.
        """

        # TODO: 2,a)
        # =================
        super(SentimentRNN, self).__init__()

        self.embedding = # TODO

        # NOTE: use batch_first=True
        self.rnn = nn.RNN(
            # TODO
        )

        self.fc = # TODO
        # =================
        # NOTE: for debugging only
        # print(self)

    def forward(self, text: torch.tensor):
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
        # TODO: 2,a)
        # =================
        embedded = # TODO
        output, hidden = # TODO

        fc_out = # TODO

        # =================
        return fc_out.squeeze(-1)


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_lstm_layers: int = 1,
    ):
        """LSTM for sentiment analysis.

        The networks consists of:
            - Embedding Layer
            - LSTM layer(s)
            - Linear layer

            Parameters
            ----------
            input_dim : int
                Input dimension of the Embebdding layer.
            embedding_dim : int
                Size of the embedding vector.
            hidden_dim : int
                The number of features in the hidden state h.
            output_dim : int
                Dimension of the output, i.e. the result of the
                linear layer.
            num_lstm_layers : int, optional
                Number of recurrent layers, by default 1.
        """
        # TODO: 2,b)
        # =================
        super(SentimentLSTM, self).__init__()

        self.embedding = # TODO

        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        # NOTE: use batch_first=True
        self.lstm = nn.LSTM(
            # TODO
        )
        self.fc = # TODO
        # =================

    def forward(self, text: torch.tensor):
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
        # TODO: 2,b)
        # =================
        embedded = # TODO
        output, (hidden, c_n) = # TODO
        fc_out = # TODO
        # =================
        return fc_out.squeeze(-1)


class DatasetIMDB:
    def __init__(
        self,
        train_to_val_ratio: float = 0.8,
        train_val_to_test_ratio: float = 0.5,
        batch_size: int = 64,
        csv_path: str = "./data/IMDB.csv",
        top_n_words: int = 1000,
        target_length: int = 500,
    ):
        """Full IMDB movie review dataset.

        Parameters
        ----------
        train_to_val_ratio : float, optional
            Ratio to split by the raw training data into training
            and validation, by default 0.8.
        train_val_to_test_ratio : float, optional
            Ratio to split by the raw dataset into training
            and testing, by default 0.5.
        batch_size : int, optional
            Batch size to use for training and validation,
            by default 64.
        csv_path : str, optional
            Path to the .csv-table containing the raw data,
            by default "./data/IMDB.csv".
        top_n_words : int, optional
            Corpus length, implemented by only taking the
            top_n_words, by default 1000.
        target_length : int, optional
            Maximum length for each review, by default 500. Shorter
            ones will be padded, longer ones will be cut off.
        """
        # TODO: 1,f)
        # =================
        data_raw = # TODO

        # separate the reviews and the labels
        input_data_raw = # TODO
        labels_raw = # TODO

        logging.info("Running prepare_data")
        input_data, labels, vocabulary = prepare_data(
            # TODO
        )
        # =================

        # save the vocabulary
        self.vocabulary = vocabulary

        # TODO: 1,g) Create the dataset with torch tensors
        # size is (50000, len(vocabulary))
        # =================
        data_all = TensorDataset(torch.from_numpy(input_data), torch.from_numpy(labels))

        len_train_full = # TODO
        len_test = # TODO

        # split the data into training and testing
        train_full, test_subset = torch.utils.data.random_split(
            # TODO
        )

        # split the train dataset again into train and validation
        len_train = # TODO
        len_val = # TODO

        train_subset, val_subset = torch.utils.data.random_split(
            # TODO
        )
        # =================

        # TODO: 1,g) Define the data loaders, pay attention to the shuffle flag
        # =================
        self.train_loader = DataLoader(
            # TODO
        )
        self.val_loader = # TODO
        self.test_loader = # TODO
        # =================

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
    # for debugging
    test = DatasetIMDB()
    print("training: ", len(test("train").dataset))
    print("validation: ", len(test("val").dataset))
    print("testing: ", len(test("test").dataset))
