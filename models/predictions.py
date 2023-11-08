# %%
import os
import pickle
import sys

sys.path.append(os.path.join(".."))

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from src.model_utils import (
    CropConvLSTM,
    CropLSTM,
    CropMLP,
    CropPL,
    CropTransformer,
    reshape_data,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import glob
from catboost import CatBoostClassifier

# %% [markdown]
# ## Paths to data

# %%
# defining paths
path_to_npys_data = os.path.join("..", "data", "npys_data")

pathFeatures = os.path.join(path_to_npys_data, "2022_2032")
pathResults = os.path.join("..", "results", "2022_2032")
pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

softmax = nn.Softmax(dim=1)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "X_down_lstm.pkl"), "rb"
) as fp:
    X_lstm = pickle.load(fp)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "X_down.pkl"), "rb"
) as fp:
    X = pickle.load(fp)

with open(
    os.path.join("..", "data", "processed_files", "pkls", "y_down.pkl"), "rb"
) as fp:
    y = pickle.load(fp)


X_test = X["Test"]
X_lstm_test = X_lstm["Test"]
y_test = y["Test"]

del X_lstm, X, y

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.long).argmax(1),
    ),
    batch_size=8192,
)
test_loader_lstm = DataLoader(
    TensorDataset(
        torch.tensor(X_lstm_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.long).argmax(1),
    ),
    batch_size=8192,
)
# %%
path_to_pickled_models = os.path.join("..", "results", "pickle_models")

clf_dict = {
    "lr": os.path.join(path_to_pickled_models, "Logistic_Regression.pkl"),
    "lgbm": os.path.join(path_to_pickled_models, "LightGBM.pkl"),
    "xgboost": os.path.join(path_to_pickled_models, "XGBoost.pkl"),
    "mlp": os.path.join(path_to_pickled_models, "MLP.ckpt"),
    "lstm": os.path.join(path_to_pickled_models, "LSTM.ckpt"),
    "transformer": os.path.join(path_to_pickled_models, "transformer.ckpt"),
    "conv_lstm": os.path.join(path_to_pickled_models, "conv_lstm.ckpt"),
    "catboost": os.path.join(path_to_pickled_models, "catboost.pkl"),
}


# %%
def make_predictions(X, clf_dict):
    """Generates predictions using different models based on provided dataset

    Args:
        X (Array or DataFrame): features

    Returns:
        y_probs: dict
        keys - model name
        values - array of probabilities
    """
    y_probs = dict()
    softmax = nn.Softmax(dim=1)
    # create an instance of pl.Trainer
    trainer = pl.Trainer(accelerator="gpu", devices=[3])
    for model in tqdm(clf_dict):
        if model == "mlp":
            network = CropMLP()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(X.values, dtype=torch.float), batch_size=8192
                    ),
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob

        elif model == "lstm":
            network = CropLSTM()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(reshape_data(X), dtype=torch.float),
                        batch_size=2048,
                    ),
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob
        elif model == "transformer":
            network = CropTransformer()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(reshape_data(X), dtype=torch.float),
                        batch_size=2048,
                    ),
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob
        elif model == "conv_lstm":
            network = CropConvLSTM()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # Make prediction
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(reshape_data(X), dtype=torch.float),
                        batch_size=2048,
                    ),
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob
        elif model == "catboost":
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(clf_dict[model])
            y_prob = loaded_model.predict_proba(X)
            y_probs[model] = y_prob
        else:
            # loading the models:
            loaded_model = pickle.load(open(clf_dict[model], "rb"))
            y_prob = loaded_model.predict_proba(X)
            y_probs[model] = y_prob

    return y_probs


# %% [markdown]
# ## Predictions of the models

use_avg = True
if use_avg:
    paths = [x for x in glob.glob(os.path.join(pathFeatures, "*.npy")) if "AVG" in x]
else:
    paths = [
        x for x in glob.glob(os.path.join(pathFeatures, "*.npy")) if "AVG" not in x
    ]

# %%
for path in tqdm(paths):  # [os.path.join(pathFeatures, "features_ssp245_AVG.npy")]
    # Loading Features
    X = pd.DataFrame.from_dict(np.load(path, allow_pickle=True), orient="columns")
    with open(
        os.path.join("..", "data", "processed_files", "pkls", "keys.pkl"), "rb"
    ) as fp:
        keys = pickle.load(fp)
    # load morf features separatly
    morf = pd.DataFrame.from_dict(
        np.load(pathMorf, allow_pickle=True), orient="columns"
    )
    X = pd.concat([X, morf], axis=1)
    # make specific order of features
    X = X[keys]
    X.replace(
        [-float("inf"), float("inf"), np.nan, -3.3999999521443642e38], 0, inplace=True
    )
    # Define scaler based on whole dataset
    scaler = joblib.load(os.path.join(path_to_npys_data, "scaler.save"))
    # Normalization using minmax scaler
    X = pd.DataFrame(scaler.transform(X), columns=keys)

    # Making predictions
    probabilities = make_predictions(X, clf_dict)

    # Saving results:
    file_name = path.split("/")[-1]  # get the file name from the path
    ssp = file_name.split("_")[1]  # get sspXXX from the file name
    geo_model = file_name.split("_")[2].split(".")[
        0
    ]  # get MRI/CNRM/CMCC from the file name

    for model in probabilities:
        with open(
            os.path.join(pathResults, "_".join([model, ssp, geo_model, "prob.npy"])),
            "wb",
        ) as f:
            pickle.dump(probabilities[model], f, protocol=4)
