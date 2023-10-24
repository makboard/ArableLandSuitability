import pickle
import os
import sys
import random
import warnings
from typing import Tuple

sys.path.append(os.path.join(".."))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from torch import nn
from torch.utils.data import DataLoader

from src.model_utils import (
    custom_multiclass_report,
    CroplandDataModule,
    CropTransformer,
    CropPL,
)

from src.evaluation import load_data

# Constants
ROOT_DIR = os.path.join("..")
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed_files", "pkls")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "results", "pickle_models")
X_DATA_FILENAME = "X_down_lstm.pkl"
Y_DATA_FILENAME = "y_down.pkl"
MODEL_FILENAME = "transformer_new.ckpt"
BATCH_SIZE = 1024
SEED = 123

def main():
    # Load data
    X, y = load_data(DATA_PATH, X_DATA_FILENAME, Y_DATA_FILENAME)

    # Initialize data module
    dm = CroplandDataModule(X=X, y=y, batch_size=BATCH_SIZE)

    # Set seeds and initialize model
    warnings.filterwarnings("ignore")
    torch.manual_seed(SEED)
    random.seed(SEED)
    network = CropTransformer()
    model = CropPL(net=network)

    # Trainer configuration
    early_stop_callback = EarlyStopping(
        monitor="val/loss", min_delta=1e-4, patience=60, verbose=True, mode="min"
    )
    model_saving = ModelCheckpoint(
        dirpath=MODEL_SAVE_PATH,
        filename=MODEL_FILENAME,
        save_top_k=3,
        mode="max",
        monitor="val/F1Score"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu",
        devices=[0],
        check_val_every_n_epoch=1,
        callbacks=[
            early_stop_callback,
            model_saving,
            lr_monitor,
            RichProgressBar(),
        ],
    )
    trainer.fit(model, dm)

    # Make predictions and evaluate
    predictions = torch.cat(
        trainer.predict(model, DataLoader(dm.X_test, batch_size=BATCH_SIZE)), dim=0
    )
    softmax = nn.Softmax(dim=1)
    yprob = softmax(predictions.float())
    ypred = torch.argmax(yprob, 1)
    ytest = torch.argmax(dm.y_test, 1).cpu().numpy()

    custom_multiclass_report(ytest, ypred, yprob)


if __name__ == "__main__":
    main()
