# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append(os.path.join(".."))

import pytorch_lightning as pl
import torch
from src.model_utils import (
    CustomWeightedRandomSampler,
    custom_multiclass_report,
    # CroplandDataModuleLSTM,
    # CropTransformer,
    CropPL,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader,Dataset


# %% [markdown]
# ### Read from file

# %%
# Read dictionary pkl file
with open(os.path.join("..", "data", "processed_files", "pkls", "X_FR.pkl"), "rb") as fp:
    X = pickle.load(fp)

with open(os.path.join("..", "data", "processed_files", "pkls", "y_FR.pkl"), "rb") as fp:
    y = pickle.load(fp)

with open(os.path.join("..", "data", "npys_data", "alpha.pkl"), "rb") as fp:
    weight = pickle.load(fp)


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


class CropTransformer(nn.Module):
    """
    A PyTorch module implementing a Crop Transformer classifier.

    The Crop_Transformer module takes as input a sequence of feature vectors and applies a Transformer encoder (https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
    followed by two linear layers with ReLU activation to predict the output.

    Args:
    input_size (int): The number of expected features in the input (default: 52).
    hidden_size (int): The number of features in the hidden state (default: 104).
    num_layers (int): Number of recurrent layers (default: 4).
    output_size (int): The number of output logits (default: 4).

    Inputs:
    X (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_size) containing the input sequence.

    Outputs:
    out (torch.Tensor): A tensor of shape (batch_size, output_size) containing the output logits.

    """

    def __init__(
        self,
        input_size=164,
        d_model=256,
        nhead=16,
        dim_feedforward=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
        activation="relu",
        output_size=4,
    ) -> None:
        super(CropTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)

        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, X):
        input = X
        embedded = self.embedding(input)
        encoded = self.transformer_enc(embedded)
        # output = encoded[:, -1, :]
        output = self.classifier(encoded)
        return output


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


# %%
# initilize data module
dm = CroplandDataModuleLSTM(X=X, y=y, batch_size=512, num_workers=0)

# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(42)

network = CropTransformer()
model = CropPL(net=network, lr=1e-3, weight=torch.FloatTensor(weight))

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss", min_delta=1e-3, patience=50, verbose=True, mode="min"
)
model_saving = ModelCheckpoint(
        save_top_k=3, mode="max", monitor="val/F1Score"
    )
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu",
    devices=[0],
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, model_saving, lr_monitor, RichProgressBar()],
)
trainer.fit(model, dm)


# %%
# Save the module to a file
model_filename = os.path.join("..", "results", "pickle_models", "transformer_FR_Xfull.pkl")
torch.save(model, model_filename)

# %%
# check metrics
predictions = torch.cat(
    trainer.predict(model, DataLoader(dm.X_test, batch_size=2048)), dim=0
)
softmax = nn.Softmax(dim=1)
yprob = softmax(predictions.float())
ypred = torch.argmax(yprob, 1)
ytest = torch.argmax(dm.y_test, 1).cpu().numpy()


print(custom_multiclass_report(ytest, ypred, yprob))
