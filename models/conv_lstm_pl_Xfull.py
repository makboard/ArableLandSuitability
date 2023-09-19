# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append('/app/ArableLandSuitability')
from src.model_utils import (CustomWeightedRandomSampler,
                            # CroplandDataModuleLSTM,
                            # CropConvLSTM,
                            CropPL,
                            custom_multiclass_report)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


# %% [markdown]
# Read dictionary pkl file
with open(os.path.join('..',
                    'data',
                    'processed_files',
                    'pkls',
                    'X_FR.pkl'), "rb") as fp:
    X = pickle.load(fp)

with open(os.path.join('..',
                    'data',
                    'processed_files',
                    'pkls',
                    'y_FR.pkl'), "rb") as fp:
    y = pickle.load(fp)


class CroplandDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        target = self.y[idx]

        return x, target


class CroplandDataModuleLSTM(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and preparing data for a Cropland classification model using LSTM architecture.

    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train, self.X_val, self.X_test = (
            torch.FloatTensor(X["Train"]),
            torch.FloatTensor(X["Val"]),
            torch.FloatTensor(X["Test"]),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.LongTensor(y["Train"]),
            torch.LongTensor(y["Val"]),
            torch.LongTensor(y["Test"]),
        )

        self.dl_dict = {"batch_size": self.batch_size, "num_workers": self.num_workers}

    def prepare_data(self):
        # Calculate class weights for imbalanced dataset
        _, counts = torch.unique(self.y_train.argmax(dim=1), return_counts=True)
        class_weights = 1.0/torch.sqrt(counts.float())
        loss_weights = class_weights/class_weights.sum()
        ds = self.y_train.argmax(dim=1)
        weights = [loss_weights[i] for i in ds]
        self.sampler = CustomWeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = CroplandDataset(
                self.X_train, self.y_train
            )
            self.dataset_val = CroplandDataset(
                self.X_val, self.y_val
            )

        if stage == "test" or stage is None:
            self.dataset_test = CroplandDataset(
                self.X_test, self.y_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            #   shuffle=True,
            pin_memory=True,
            sampler=self.sampler,
            **self.dl_dict,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, pin_memory=True, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, pin_memory=True, **self.dl_dict)


class ConvLSTMCell(nn.Module):
    """
    Initialize ConvLSTM cell.

    Args:
    input_dim (int): Number of channels of input tensor.
    hidden_dim (int): Number of channels of hidden state.
    kernel_size (int): Size of the convolutional kernel.
    bias (bool): Whether to add the bias.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias

        self.conv = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, length):
        return (
            torch.zeros(
                batch_size, self.hidden_dim, length, device=self.conv.weight.device
            ),
            torch.zeros(
                batch_size, self.hidden_dim, length, device=self.conv.weight.device
            ),
        )


class CropConvLSTM(nn.Module):

    """
    A PyTorch module implementing a Crop Conv LSTM network.

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        n_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers

    Input:
        A tensor of size B, T, C
    Output:
        A tuple of two lists of length n_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple,
        n_layers: int,
        n_classes: int,
        input_len_monthly: int,
        seq_len: int,
        # input_len_static: int,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super(CropConvLSTM, self).__init__()

        assert isinstance(kernel_size, tuple) or (
            isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size])
        ), "`kernel_size` must be tuple or list of tuples"
        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == n_layers
        kernel_size = self._extend_for_multilayer(kernel_size, n_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, n_layers)
        assert (
            len(kernel_size) == len(hidden_dim) == n_layers
        ), "Inconsistent list length."

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.n_classes = n_classes
        self.input_len_monthly = input_len_monthly
        self.seq_len = seq_len
        # self.input_len_static = input_len_static

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(
                self.hidden_dim[0] * self.seq_len * self.input_len_monthly,
                # + self.input_len_static,
                self.hidden_dim[0] * self.seq_len,
            ),
            nn.BatchNorm1d(self.hidden_dim[0] * self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim[0] * self.seq_len, self.seq_len),
            nn.BatchNorm1d(self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.seq_len, self.n_classes),
        )

    def forward(self, input, hidden_state=None):
        """
        Args:
        input (tuple):  input[0] is a tensor of shape (batch_size, sequence_length, input_size) containing the monthly input sequence.
                        input[1] is a tensor of shape (batch_size, input_size) containing the static input sequence.

        Returns
        -------
        last_state_list, layer_output
        """
        input_monthly = input[
            :, None, :, :
        ]  # fictional dimension added. will be used as channnels
        # input_static = input[1]
        b = input_monthly.size()[0]

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=b, length=self.input_len_monthly
            )

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_monthly

        for layer_idx in range(self.n_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.seq_len):
                h, c = self.cell_list[layer_idx](
                    input=cur_layer_input[:, :, t, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        output_monthly = self.flatten(layer_output_list[0])
        output = self.net(output_monthly)

        return output

    def _init_hidden(self, batch_size, length):
        init_states = []
        for i in range(self.n_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, length))
        return init_states

    # @staticmethod
    # def _check_kernel_size_consistency(kernel_size):
    #     if not (isinstance(kernel_size, tuple) or
    #             (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
    #         raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, n_layers):
        if not isinstance(param, list):
            param = [param] * n_layers
        return param



# initilize data module
dm = CroplandDataModuleLSTM(X=X, y=y, batch_size=1024)


# %% [markdown]
# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(123)
random.seed(123)
            
network = CropConvLSTM(
    input_dim=1, #fictional dimension. will be used as channnels
    hidden_dim=16,
    kernel_size=(3,),
    n_layers=1,
    n_classes = 4,
    input_len_monthly = X['Train'].shape[2],
    seq_len = X['Train'].shape[1],
    # input_len_static = X['Train'].shape[1],
    bias=False,
    return_all_layers=False
    )

model = CropPL(net=network)

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss",
    min_delta=1e-4, patience=30, verbose=True, mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    precision=16,
    devices=[1],
    benchmark=True,
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, lr_monitor],
)
trainer.fit(model, dm)


# %% [markdown]
class CroplandDataset_test(Dataset):
    def __init__(self, X):
        self.X_monthly = X[0]  
        self.X_static = X[1] 

    def __len__(self):
        return len(self.X_monthly)

    def __getitem__(self, idx):
        x_monthly = self.X_monthly[idx]
        x_static = self.X_static[idx]

        return (x_monthly, x_static)


# %%  
# check metrics
# predictions = torch.cat(trainer.predict(model, DataLoader(CroplandDataset_test((dm.X_monthly_test, dm.X_static_test)),
#                                         batch_size=2048)), dim=0)
# softmax = nn.Softmax(dim=1)
# yprob = softmax(predictions.float())
# ypred = torch.argmax(yprob, 1)
# ytest = torch.argmax(dm.y_test, 1).cpu().numpy()


# print(custom_multiclass_report(ytest, ypred, yprob))


# %%
# Save the module to a file
model_filename = os.path.join("..", "results", "pickle_models", "conv_lstm_Xfull.pkl")
torch.save(model, model_filename)