import argparse
import os
import sys
import pickle
from typing import List
from typing import Optional

sys.path.append(os.path.join(".."))

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

BATCHSIZE = 1024
CLASSES = 4
EPOCHS = 500
DIR = os.getcwd()

# Read dictionary pkl file
with open(os.path.join('data', 'processed_files', 'pkls', 'X_down.pkl'), "rb") as fp:
    X = pickle.load(fp)

with open(os.path.join('data', 'processed_files', 'pkls', 'y_down.pkl'), "rb") as fp:
    y = pickle.load(fp)


class Net(nn.Module):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 128
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy, sync_dist=True)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
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

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = TensorDataset(self.X_test, self.y_test)
    
        self.mnist_train = TensorDataset(self.X_train, self.y_train)
        
        self.mnist_val = TensorDataset(self.X_val, self.y_val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
    


def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 20)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 512, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
    callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        devices=3,
        callbacks=[callback],
        strategy="ddp_spawn",
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    callback.check_pruned()

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning distributed data-parallel training example."
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    
    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))