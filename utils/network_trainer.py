# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.decorators import log_time


class NetworkTrainer:
    def __init__(
        self,
        network_class: nn.Module,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        data_loader: DataLoader,
        folder: str = "./metadata/",
    ):
        """Construct a network trainer, to generically train arbitrary networks.

        Parameters
        ----------
        network_class : nn.Module
            Class of the network, e.g. ``DNN``.
        loss_function : nn.Module
            Loss function, e.g. ``torch.nn.CrossEntropyLoss``.
        optimizer : torch.optim.Optimizer
            Optimizer to minimize the loss, e.g. ``torch.optim.Adam``.
        lr_scheduler : torch.optim.lr_scheduler
            Step size scheduler to reduce the step size based on
            different rule, e.g. ``torch.optim.lr_scheduler.ReduceLROnPlateau``.
        data_loader : torch.utils.data.DataLoader
            Data loader to provide the train and validation dataset, must be callable
            via
            ```python
            >>> train_data = data_loader("train")
            >>> validation_data = data_loader("validation")
            ```
            to get the data.
        folder : str, optional
            Folder to store meta information, by default "./metadata/".
        """
        self.network_class = network_class
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader

        # folder is used to save the model parameters and loss-logs
        self.folder = folder

    # decorator to stop the training time
    @log_time
    def train_network(
        self,
        num_epochs: int,
        network_params: dict,
        loss_params: dict,
        optimizer_params: dict,
        scheduler_params: dict,
        sum_correct_preds: callable = lambda pred_output, target_output: -1,
        regularization: callable = lambda model: 0,
    ):
        """Gym: Train a network for num_epochs epochs given the network's
        parameters and the data given by the constructor.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train your network.
        network_params : dict
            Parameters for the network written as a dict. If your network header looks
            like:
            ```python
            >>> def __init__(
            >>>     self,
            >>>     input_shape: int,
            >>>     hidden_shape: int,
            >>>     num_classes: int,
            >>>     activation_func,
            >>>     **kwargs,
            >>> ):
            ```
            your dict should look like:
            ```python
            >>> network_params = {
            >>>     "input_shape": 28 * 28,
            >>>     "hidden_shape": 500,
            >>>     "num_classes": 10,
            >>>     "activation_func": nn.ReLU,
            >>> }
            ```
        loss_params : dict
            Parameters used for the loss function, works similar as the network_params.
        optimizer_params : dict
            Parameters used for the optimizer, works similar as the network_params.
        scheduler_params : dict
            Parameters used for the scheduler, works similar as the network_params.
        sum_correct_preds : callable
            Function which is called after every batch iteration in the train and the
            validation cycle and is used to manually compute the accuracy of a model.
            The function always has the parameters ``prediction_output`` and
            ``target_output`` both beeing of the type ``torch.tensor``.
            The default is the const. function -1 and thus, without a callback the accuracy
            is always negative.
            Given a prediction and a target, sum_correct_preds computes the absolute
            number of correct predictions. Internally the sum of all those values is
            averaged and used to compute the accuracy in percentage points.
        regularization : callable
            Used to add regularization (e.g. L1, L2) to the loss function.
            ``regularization`` is called after every loss computation with the
            ``model`` as a parameter and the return value is added to the value
            of the loss function. The default is the zero function.

        Returns
        -------
        str
            Path to the saved parameters and the metadata of the training process
            (np.array with training and validation loss for each epoch).
        """

        model = self.network_class(**network_params)
        criterion = self.loss_function(**loss_params)

        optimizer = self.optimizer(model.parameters(), **optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **scheduler_params)

        def _scheduler_step(avg_val_loss: float):
            """Some schedulers need the error (e.g. ReduceLROnPlateau) others don't."""
            try:
                scheduler.step(avg_val_loss)
            except:
                scheduler.step()

        train_dataset = self.data_loader("train")
        validation_dataset = self.data_loader("val")

        len_train_data = len(train_dataset.dataset)
        len_val_data = len(validation_dataset.dataset)

        self.metadata = np.zeros((num_epochs, 2))

        for epoch in range(num_epochs):

            print("EPOCH {}/{}".format(epoch + 1, num_epochs))
            # TODO: 3,a) Define two variables train_loss and validation_loss initialized
            # with the values 0, you can see in the code below how they are going to be used
            # ============
            train_loss = 0
            validation_loss = 0
            # ============

            # TRAIN
            model.train()

            train_correct = 0
            for batch_idx, (*input_data, target_output) in enumerate(
                tqdm(train_dataset, "Training")
            ):
                # TODO: 3,b) Take a look at the constructor's documentation for a detailed
                # description of the different components (in total ~5 lines of code)
                # ============
                optimizer.zero_grad()
                predicted_output = model(*input_data)
                loss = criterion(predicted_output, target_output)

                loss += regularization(model)

                # backward propagation step + optimizer step
                loss.backward()
                optimizer.step()
                # ============

                # TODO: 3,c)
                # Return the actual loss value by calling .item() --> gives a scalar value
                # ============
                train_loss += loss.item()
                # ============

                train_correct += sum_correct_preds(predicted_output, target_output)

            # TODO: 3,c)
            # Compute the average train loss, len_train_data is the number of mini batches we use
            # ============
            train_loss /= len_train_data
            # ============

            # VALIDATION
            model.eval()
            val_correct = 0
            # TODO: 3,d)
            # We turn off backpropagation as we don't want to perform a backpropagation step
            # By using ``torch.no_grad()`` we disable the backward pass in the computational graph
            # to save time and reduce our memory footprint
            # ============
            with torch.no_grad():
                for batch_idx, (*input_data, target_output) in enumerate(
                    tqdm(validation_dataset, "Validation")
                ):
                    predicted_output = model(*input_data)
                    loss = criterion(predicted_output, target_output)
                    validation_loss += loss.item()

                    val_correct += sum_correct_preds(predicted_output, target_output)

            validation_loss /= len_val_data
            # ============

            # show results of the training
            print(
                "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    train_loss,
                    train_correct,
                    len_train_data,
                    100.0 * train_correct / len_train_data,
                )
            )
            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    validation_loss,
                    val_correct,
                    len_val_data,
                    100.0 * val_correct / len_val_data,
                )
            )
            self._log_epoch(epoch, train_loss, validation_loss)

            # STEP SIZE SCHEDULER
            _scheduler_step(validation_loss / len_val_data)

        # OUT OF THE LOOP: save model weights and metadata
        return self._save(model.state_dict(), network_params)

    def _save(self, state_dict: dict, network_params: dict):
        """Save model params and metadata."""
        filename = self._generate_filename(self.network_class, network_params)
        full_path = self.folder + filename
        torch.save(state_dict, full_path + "_model.pt")
        np.save(full_path + "_loss.npy", self.metadata)
        return filename

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Helper function to extend the metadata."""
        self.metadata[epoch, :] = np.array([train_loss, val_loss])

    def _generate_filename(self, network_class: nn.Module, network_params: dict):
        """Helper function to generate a unique filename based on the network's parameters"""
        filename = ""
        for key, value in network_params.items():
            filename += "{}_{}_".format(key, value)
        filename += str(network_class)
        return self._remove_special_chars(filename)

    def _remove_special_chars(self, filename: str) -> str:
        """Remove special characters for windows filenames."""
        remove_char = ["<", ">", ":", '"', "/", "", "|", "?", "*"]
        return "".join(["" if i in remove_char else i for i in filename])
