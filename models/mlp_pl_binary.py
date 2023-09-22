# %%
import os
import pickle
import random
import sys
import warnings

sys.path.append(os.path.join("/app/ArableLandSuitability/"))

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
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

# %% [markdown]
# ### Read from file

# %%
# Read dictionary pkl file
with open(
    os.path.join(
        "/app/ArableLandSuitability/", "data", "processed_files", "pkls", "X_FR.pkl"
    ),
    "rb",
) as fp:
    X = pickle.load(fp)

with open(
    os.path.join(
        "/app/ArableLandSuitability/", "data", "processed_files", "pkls", "y_FR.pkl"
    ),
    "rb",
) as fp:
    y = pickle.load(fp)

with open(
    os.path.join("/app/ArableLandSuitability/", "data", "npys_data", "alpha.pkl"), "rb"
) as fp:
    weight = pickle.load(fp)

# %%
# initilize data module
dm = CroplandDataModuleMLP(X=X, y=y, batch_size=16834, num_workers=0, num_classes=2)

# initilize model
warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(42)

experiment_name = "MLP_binary_2048"
num_classes = 2
network = CropMLP(output_size=num_classes)
network.initialize_bias_weights(dm.y_train.argmax(dim=1))
model = CropPL(net=network, lr=1e-3, num_classes=num_classes, weight=None)

# initilize trainer
early_stop_callback = EarlyStopping(
    monitor="val/loss", min_delta=1e-3, patience=50, verbose=True, mode="min"
)
model_saving = ModelCheckpoint(save_top_k=3, mode="max", monitor="val/F1Score")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu",
    devices=[0],
    check_val_every_n_epoch=1,
    default_root_dir="/app/ArableLandSuitability/models/lightning_logs/"
    + experiment_name,
    callbacks=[early_stop_callback, model_saving, lr_monitor, RichProgressBar()],
)
trainer.fit(model, dm)


# %%
# check metrics
dm.setup()
predict_output = (
    trainer.predict(
        model,
        DataLoader(dm.dataset_test, batch_size=2048),
        ckpt_path="/app/ArableLandSuitability/models/lightning_logs/MLP_binary_2048/lightning_logs/version_1/checkpoints/epoch=99-step=66800.ckpt",
    ),
)
preds_tuple, tgts_tuple = zip(*predict_output[0])
predictions = torch.cat(
    preds_tuple,
    dim=0,
)
path_preds = "/app/ArableLandSuitability/results/predictions/" + experiment_name
if not os.path.exists(path_preds):
    os.makedirs(path_preds)
torch.save(predictions, os.path.join(path_preds, "preds.pt"))

# %%
# Save the module to a file
model_filename = os.path.join(
    "/app/ArableLandSuitability/", "results", "pickle_models", experiment_name + ".pkl"
)
torch.save(model, model_filename)

softmax = nn.Softmax()
preds = softmax(predictions)[:, 1].detach().cpu().numpy()
tgt = predictions = (
    torch.cat(
        tgts_tuple,
        dim=0,
    )
    .argmax(dim=1)
    .detach()
    .cpu()
    .numpy()
)
precision, recall, thresholds = precision_recall_curve(tgt, preds)
hmeans = 2 / (1 / recall[:-1] + 1 / precision[:-1])
th = thresholds[hmeans.argmax()]
print("ROCAUC", roc_auc_score(tgt, preds))
print("AP", average_precision_score(tgt, preds))
print("Accuracy", accuracy_score(tgt, preds > th))
print("Precision", precision_score(tgt, preds > th))
print("Recall", recall_score(tgt, preds > th))
print("f1", f1_score(tgt, preds > th))
pass
