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
    custom_multiclass_report,
    CroplandDataModuleMLP,
    CropMLP,
    CropPL,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader

# %% [markdown]
# ### Read from file

# %%
# Read dictionary pkl file
with open(
    os.path.join("..", "data", "processed_files", "pkls", "X_FR.pkl"), "rb"
) as fp:
    X = pickle.load(fp)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "y_FR.pkl"), "rb"
) as fp:
    y = pickle.load(fp)

with open(os.path.join("..", "data", "npys_data", "alpha.pkl"), "rb") as fp:
    weight = pickle.load(fp)

# %%
# initilize data module
dm = CroplandDataModuleMLP(X=X, y=y, batch_size=8192, num_workers=0)

# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(142)
random.seed(142)

network = CropMLP()
network.initialize_bias_weights(dm.y_train.argmax(dim=1))
model = CropPL(net=network, lr=1e-3, weight=None)

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss", min_delta=1e-3, patience=50, verbose=True, mode="min"
)
model_saving = ModelCheckpoint(save_top_k=3, mode="max", monitor="val/F1Score")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu",
    devices=[2],
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, model_saving, lr_monitor, RichProgressBar()],
)
trainer.fit(model, dm)


# %%
# check metrics
predictions = torch.cat(
    trainer.predict(model, DataLoader(dm.X_test, batch_size=16384)), dim=0
)
softmax = nn.Softmax(dim=1)
yprob = softmax(predictions.float())
ypred = torch.argmax(yprob, 1)
ytest = torch.argmax(dm.y_test, 1).cpu().numpy()


print(custom_multiclass_report(ytest, ypred, yprob))

# %%
# Save the module to a file
model_filename = os.path.join("..", "results", "pickle_models", "mlp_FR.pkl")
torch.save(model, model_filename)
