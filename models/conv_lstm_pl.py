# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append('/app/ArableLandSuitability')
from src.model_utils import (CroplandDataModuleLSTM,
                            CropConvLSTM,
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
                    'X_FR_ROS_lstm.pkl'), "rb") as fp:
    X = pickle.load(fp)

with open(os.path.join('..',
                    'data',
                    'processed_files',
                    'pkls',
                    'y_FR_ROS_lstm.pkl'), "rb") as fp:
    y = pickle.load(fp)

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
    input_len_monthly = X['Train'][0].shape[2],
    seq_len = X['Train'][0].shape[1],
    input_len_static = X['Train'][1].shape[1],
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
model_filename = os.path.join("..", "results", "pickle_models", "conv_lstm_FR_ROS.pkl")
torch.save(model, model_filename)