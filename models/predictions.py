# %%
import os
import pickle
import sys

sys.path.append(os.path.join(".."))

import gc
import glob
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from src.model_utils import Crop_LSTM, Crop_MLP, Crop_PL, reshape_data
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# %% [markdown]
# ## Paths to data

# %%
# defining paths
path_to_npys_data = os.path.join("..", "data", "npys_data")

pathFeatures = os.path.join(path_to_npys_data, "2040_2050")
pathResults = os.path.join("..", "results", "2040_2050")
pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

softmax = nn.Softmax()

# %%
path_to_pickled_models = os.path.join("..", "results", "pickle_models")

clf_dict = {
    "lr": os.path.join(
        path_to_pickled_models, "Logistic_Regression_crops_final.pkl"
    ),
    "xgbt": os.path.join(path_to_pickled_models, "XGBoost_crops_final.pkl"),
    "lgbm": os.path.join(path_to_pickled_models, "LightGBM_crops_final.pkl"),
    "MLP": os.path.join(path_to_pickled_models, "Crop_MLP.ckpt"),
    "lstm": os.path.join(path_to_pickled_models, "Crop_LSTM.ckpt"),
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

    for model in tqdm(clf_dict):
        if model == "MLP":
            network = Crop_MLP()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = Crop_PL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # create an instance of pl.Trainer
            trainer = pl.Trainer(gpus=1)

            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(X.values, dtype=torch.float), batch_size=2048
                    ),
                ),
                dim=0,
            )
            softmax = nn.Softmax(dim=1)
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob

        elif model == "lstm":
            network = Crop_LSTM()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = Crop_PL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # create an instance of pl.Trainer
            trainer = pl.Trainer(gpus=1)

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
            softmax = nn.Softmax(dim=1)
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob

        else:
            # loading the models:
            loaded_model = pickle.load(open(clf_dict[model], "rb"))
            y_prob = loaded_model.predict_proba(X)
            y_probs[model] = y_prob

    return y_probs


# %% [markdown]
# ## Predictions of the models

# %%
for path in tqdm(glob.glob(os.path.join(pathFeatures, "*.npy"))):
    # Loading Features
    X = pd.DataFrame.from_dict(np.load(path, allow_pickle=True), orient="columns")
    with open(os.path.join(path_to_npys_data, "keys.pkl"), "rb") as fp:
        keys = pickle.load(fp)
    # load morf features separatly
    morf = pd.DataFrame.from_dict(
        np.load(pathMorf, allow_pickle=True), orient="columns"
    )
    X = pd.concat([X, morf], axis=1)
    # make specific order of features
    X = X[keys]
    X = X.replace(-np.inf, 0)
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


# %% [markdown]
# ## Average probability based on different climate models

# %%
cmcc = np.load(
    os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_CMCC_prob.npy"), allow_pickle=True
)
cnrm = np.load(
    os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_CNRM_prob.npy"), allow_pickle=True
)
mri = np.load(
    os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_MRI_prob.npy"), allow_pickle=True
)

average = np.mean([cmcc, cnrm, mri], axis=0)
with open(
    os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_average_prob.npy"),
    "wb",
) as f:
    pickle.dump(average, f, protocol=4)
