# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append(os.path.join(".."))

import pytorch_lightning as pl
import torch
from src.model_utils import custom_multiclass_report, CroplandDataModule_MLP
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics


# %%
class Crop_MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) used for crop classification.

    Args:
        input_size (int): The number of input features (default: 162).
        output_size (int): The number of output logits (default: 4).

    Inputs:
        X (torch.Tensor): A tensor of shape (batch_size, input_size) containing input data.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, output_size) containing the output logits.
    """

    def __init__(self, input_size=162, output_size=4) -> None:
        super(Crop_MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 8 * input_size),
            nn.BatchNorm1d(8 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(8 * input_size, 4 * input_size),
            nn.BatchNorm1d(4 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * input_size, 2 * input_size),
            nn.BatchNorm1d(2 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size // 2, input_size // 8),
            nn.BatchNorm1d(input_size // 8),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size // 8, output_size),
        )

        # self.initialize_weights()

    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output

    # def initialize_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             torch.nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 torch.nn.init.constant_(module.bias, 0.0)


# %% [markdown]
# ### Read from file

# %%
# Read dictionary pkl file
with open(
    os.path.join("..", "data", "processed_files", "pkls", "X_TEST1.pkl"), "rb"
) as fp:
    X = pickle.load(fp)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "y_TEST1.pkl"), "rb"
) as fp:
    y = pickle.load(fp)


# %%
class Crop_PL(pl.LightningModule):
    """
    PyTorch Lightning module for training a crop classification neural network.

    Args:
    net (torch.nn.Module): the neural network module to be trained.
    num_classes

    Attributes:
    softmax (nn.Softmax): softmax activation function.
    criterion (nn.CrossEntropyLoss): cross entropy loss function.
    optimizer (torch.optim.Adam)
    scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes=4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.softmax = nn.Softmax()
        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.val_F1Score_best = torchmetrics.MaxMetric()

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_F1Score_best.reset()

    def model_step(self, batch):
        objs, target = batch
        predictions = self(objs)
        loss = self.loss(predictions, target.float())
        return loss, self.softmax(predictions), torch.argmax(target, dim=1)

    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.train_loss(loss)
        self.train_accuracy(predictions, target)
        self.train_recall(predictions, target)
        self.train_precision(predictions, target)
        self.train_F1Score(predictions, target)
        self.train_avg_precision(predictions, target)

        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log(
            "train/F1Score",
            self.train_F1Score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train/AP", self.train_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def validation_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.val_loss(loss)
        self.val_accuracy(predictions, target)
        self.val_recall(predictions, target)
        self.val_precision(predictions, target)
        self.val_F1Score(predictions, target)
        self.val_avg_precision(predictions, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log(
            "val/F1Score", self.val_F1Score, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/AP", self.val_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def validation_epoch_end(self, outputs):
        F1Score = self.val_F1Score.compute()
        self.val_F1Score_best(F1Score)
        self.log("val/F1Score_best", self.val_F1Score_best.compute(), prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_accuracy(predictions, target)
        self.test_recall(predictions, target)
        self.test_precision(predictions, target)
        self.test_F1Score(predictions, target)
        self.test_avg_precision(predictions, target)

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log(
            "test/F1Score",
            self.test_F1Score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/AP", self.test_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        if scheduler is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                patience=20,
                mode="min",
                factor=0.5,
                verbose=True,
                min_lr=1e-8,
                threshold=5e-4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# %%
# initilize data module
dm = CroplandDataModule_MLP(X=X, y=y, batch_size=256)

# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(123)
random.seed(123)

network = Crop_MLP(output_size=4)
model = Crop_PL(net=network, num_classes=4)
# Load the checkpoint and create the model
checkpoint = torch.load(os.path.join("lightning_logs", "version_102", "checkpoints", "epoch=16-step=887570.ckpt"))
model = Crop_PL(net=network, num_classes=4)

# Get the state dict of the checkpoint and remove the parameters you want to exclude
state_dict = checkpoint['state_dict']

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss", min_delta=1e-4, patience=60, verbose=True, mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=10000,
    gpus=1,
    # precision=16,
    benchmark=True,
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, lr_monitor],
)
trainer.fit(model, dm)


# %%
from model_utils import custom_multiclass_report

# check metrics
predictions = torch.cat(
    trainer.predict(model, DataLoader(dm.X_test, batch_size=256)), dim=0
)
softmax = nn.Softmax(dim=1)
yprob = softmax(predictions.float())
ypred = torch.argmax(yprob, 1)
ytest = torch.argmax(dm.y_test, 1).cpu().numpy()


print(custom_multiclass_report(ytest, ypred, yprob))

# %%
# Save the module to a file
torch.save(model, "lightning_logs/version_104/my_module.pth")
