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

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # NOTE: use batch_first=True
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
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
        # print("forward")
        # print(type(text))
        # print(text.shape)
        # embedded shape:
        # torch.Size([64, 500, 64])
        # batch_size, num_words, embedding_dim

        # TODO: 2,a)
        # =================
        embedded = self.embedding(text)
        # print(embedded.shape)

        # shape output:
        # torch.Size([64, 500, 256])
        # batch_size, num_words, hidden_dim

        # shape hidden:
        # torch.Size([1, 64, 256]) --> squeeze is needed
        # _, batch_size, hidden_dim
        output, hidden = self.rnn(embedded)  # , h0.detach())
        # print("output: ", output.shape)
        # print("hidden: ", hidden.shape)
        # print("hidden.squeeze", hidden.squeeze(0).shape)

        # NOTE: for debugging reasons only
        # assert torch.equal(output[:, -1, :], hidden.squeeze(0))

        # shape fc_out
        # torch.Size([64, 1]) --> squeeze is needed
        # batch_size, num_layers(=1)
        fc_out = self.fc(hidden.squeeze(0))

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

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        # NOTE: use batch_first=True
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
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
        embedded = self.embedding(text)
        output, (hidden, c_n) = self.lstm(embedded)
        fc_out = self.fc(hidden.squeeze(0))
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
        data_raw = pd.read_csv(csv_path)

        # separate the reviews and the labels
        input_data_raw = data_raw["review"].values
        labels_raw = data_raw["sentiment"].values

        logging.info("Running prepare_data")
        input_data, labels, vocabulary = prepare_data(
            input_data_raw,
            labels_raw,
            top_n_words=top_n_words,
            target_length=target_length,
        )
        # =================

        # save the vocabulary
        self.vocabulary = vocabulary

        # TODO: 1,g) Create the dataset with torch tensors
        # size is (50000, len(vocabulary))
        # =================
        data_all = TensorDataset(torch.from_numpy(input_data), torch.from_numpy(labels))

        len_train_full = int(len(data_all) * train_val_to_test_ratio)
        len_test = len(data_all) - len_train_full

        # split the data into training and testing
        train_full, test_subset = torch.utils.data.random_split(
            data_all, [len_train_full, len_test]
        )

        # split the train dataset again into train and validation
        len_train = int(len(train_full) * train_to_val_ratio)
        len_val = len(train_full) - len_train

        train_subset, val_subset = torch.utils.data.random_split(
            train_full, [len_train, len_val]
        )
        # =================

        # TODO: 1,g) Define the data loaders
        # =================
        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
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
