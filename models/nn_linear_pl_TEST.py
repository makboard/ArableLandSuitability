# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append(os.path.join(".."))

import pytorch_lightning as pl
import torch
from src.model_utils import custom_multiclass_report, CroplandDataModule_MLP, Crop_MLP, Crop_PL
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision('medium')


# %% [markdown]
# ### Read from file

# %%
# Read dictionary pkl file
with open(
    os.path.join("..", "data", "processed_files", "pkls", "X.pkl"), "rb"
) as fp:
    X = pickle.load(fp)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "y.pkl"), "rb"
) as fp:
    y = pickle.load(fp)


# %%
# initilize data module
# initilize data module
dm = CroplandDataModule_MLP(X=X, y=y, batch_size=1024)

# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(42)

network = Crop_MLP()
network.initialize_bias_weights(dm.y_train.argmax(dim=1))
model = Crop_PL(net=network)

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss", min_delta=5e-4, patience=20, verbose=True, mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=500,
    accelerator='gpu',
    precision=16,
    devices=[3],
    benchmark=True,
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, lr_monitor],
)
trainer.fit(model, dm)


# %%
from src.model_utils import custom_multiclass_report

# check metrics
predictions = torch.cat(trainer.predict(model, DataLoader(dm.X_test, batch_size=2048)), dim=0)
softmax = nn.Softmax(dim=1)
yprob = softmax(predictions.float())
ypred = torch.argmax(yprob, 1)
ytest = torch.argmax(dm.y_test, 1).cpu().numpy()


print(custom_multiclass_report(ytest, ypred, yprob))