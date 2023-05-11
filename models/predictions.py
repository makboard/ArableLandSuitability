# %%
import os
import pickle
import sys

sys.path.append(os.path.join(".."))

import gc
import os
import pickle
import joblib
import glob

import numpy as np
import pandas as pd
import torch
import xgboost
import lightgbm
from model_utils import reshape_data, drop_classes, Crop_MLP, Crop_LSTM, Crop_PL
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from temperature_scaling import ModelWithTemperature


# %% [markdown]
# ## Paths to data

# %%
# defining paths
path_to_Npys_data = os.path.join("..", "data", "Npys_data")

pathFeatures = os.path.join(path_to_Npys_data, "2022_2032")
pathResults = os.path.join("..", "results", "2022_2032")
pathMorf = os.path.join(path_to_Npys_data, "features_morf_data.npy")

softmax = nn.Softmax()

# %%
path_to_pickled_models = os.path.join("..", "results", "pickle_models")

clf_dict = {
    # "lr": os.path.join(
    #     path_to_pickled_models, "Logistic_Regression_3_avg_final_v1.pkl"
    # ),
    # "xgbt": os.path.join(path_to_pickled_models, "XGBoost_crops_final.pkl"),
    # "lgbm": os.path.join(path_to_pickled_models, "LightGBM_3_avg_final_v1.pkl"),
    # "MLP": os.path.join(path_to_pickled_models, "Crop_MLP.ckpt"),
    "lstm": os.path.join(path_to_pickled_models, "Crop_LSTM.ckpt"),
}

# Features
with open(os.path.join("..", "data", "Processed_Files", "X_down_all.pkl"), "rb") as fp:
    X = pickle.load(fp)

# Target Variable
with open(os.path.join("..", "data", "Processed_Files", "y_down_all.pkl"), "rb") as fp:
    y = pickle.load(fp)

with open(os.path.join(path_to_Npys_data, "keys.pkl"), "rb") as fp:
        keys = pickle.load(fp)
        
X_val = pd.DataFrame(X['Val'], columns=keys)
y_val = y['Val']

del X, y

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
            loaded_model.load_state_dict(checkpoint['state_dict'])
            loaded_model.eval()
            scaled_model = ModelWithTemperature(loaded_model.net)
            # Create a TensorDataset that combines X_val and y_val
            dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float), torch.tensor(y_val, dtype=torch.long).argmax(1))
            # Pass the dataset to your DataLoader, along with any other relevant arguments
            valid_loader = DataLoader(dataset, batch_size=2048)
            scaled_model.set_temperature(valid_loader)
            scaled_model = Crop_PL(net=scaled_model)
            
            # create an instance of pl.Trainer
            trainer = pl.Trainer(accelerator='gpu', devices=1)

            # check metrics
            predictions = torch.cat(trainer.predict(scaled_model, DataLoader(torch.tensor(X.values, dtype=torch.float), batch_size=2048)), dim=0)
            softmax = nn.Softmax(dim=1)
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob
            
        elif model == "lstm":
            network = Crop_LSTM()
            checkpoint = torch.load(clf_dict[model])
            loaded_model = Crop_PL(net=network)
            loaded_model.load_state_dict(checkpoint['state_dict'])
            loaded_model.eval()
            scaled_model = ModelWithTemperature(loaded_model.net)
            # Create a TensorDataset that combines X_val and y_val
            dataset = TensorDataset(torch.tensor(reshape_data(X_val), dtype=torch.float), torch.tensor(y_val, dtype=torch.long).argmax(1))
            # Pass the dataset to your DataLoader, along with any other relevant arguments
            valid_loader = DataLoader(dataset, batch_size=2048)
            scaled_model.set_temperature(valid_loader)
            scaled_model = Crop_PL(net=scaled_model)

            # create an instance of pl.Trainer
            trainer = pl.Trainer(accelerator='gpu', devices=1)

            # check metrics
            predictions = torch.cat(trainer.predict(scaled_model, DataLoader(torch.tensor(reshape_data(X), dtype=torch.float), batch_size=2048)), dim=0)
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
    with open(os.path.join(path_to_Npys_data, "keys.pkl"), "rb") as fp:
        keys = pickle.load(fp)
    # load morf features separatly
    if len(X.keys())<160:
        morf = pd.DataFrame.from_dict(
            np.load(pathMorf, allow_pickle=True), orient="columns"
        )
        X = pd.concat([X, morf], axis=1)
    # make specific order of features
    X = X[keys]
    X = X.replace(-np.inf, 0)
    # Define scaler based on whole dataset
    scaler = joblib.load(os.path.join(path_to_Npys_data, "scaler.save"))
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
