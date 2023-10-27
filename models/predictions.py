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
from src.model_utils import reshape_data, CroplandDatasetTest
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from catboost import CatBoostClassifier

## Paths to data
path_to_npys_data = os.path.join("..", "data", "npys_data")

pathFeatures = os.path.join(path_to_npys_data, "2022_2032")
pathResults = os.path.join("..", "data", "results", "2022_2032")
pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

softmax = nn.Softmax(dim=1)

path_to_pickled_models = os.path.join("..", "results", "pickle_models")

clf_dict = {
    # "lr": os.path.join(
    #     path_to_pickled_models, "Logistic_Regression_crops_final.pkl"
    # ),
    "catboost": os.path.join(path_to_pickled_models, "catboost.pkl"),
    "transformer": os.path.join(path_to_pickled_models, "transformer.pkl"),
    "mlp": os.path.join(path_to_pickled_models, "mlp.pkl"),
    "lstm": os.path.join(path_to_pickled_models, "lstm.pkl"),
    "conv_lstm": os.path.join(path_to_pickled_models, "conv_lstm.pkl"),
}

def make_predictions(X, X_keys, clf_dict):
    """Generates predictions using different models based on provided dataset

    Args:
        X (Array or DataFrame): features

    Returns:
        y_probs: dict
        keys - model name
        values - array of probabilities
    """
    y_probs = dict()
    trainer = pl.Trainer(accelerator="gpu", devices=[0])
    
    for model in tqdm(clf_dict):
        if model == "mlp":
            loaded_model = torch.load(clf_dict[model])
            loaded_model.eval()

            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        torch.tensor(X.values, dtype=torch.float), batch_size=8192 * 4
                    ),
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob

        elif model in ["lstm", "conv_lstm", "transformer"]:
            loaded_model = torch.load(clf_dict[model])
            loaded_model.eval()
            X_monthly, X_static, _, _ = reshape_data(pd.DataFrame(X, columns=X_keys))
            X_monthly = torch.tensor(X_monthly, dtype=torch.float)
            X_static = torch.tensor(X_static, dtype=torch.float)
            # check metrics
            predictions = torch.cat(
                trainer.predict(
                    loaded_model,
                    DataLoader(
                        CroplandDatasetTest((X_monthly, X_static)),
                        batch_size=8192 * 4,
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


## Predictions of the models
for path in tqdm(glob.glob(os.path.join(pathFeatures, "*.npy"))):
    # Loading Features
    X = pd.DataFrame.from_dict(np.load(path, allow_pickle=True), orient="columns")
    with open(os.path.join(path_to_npys_data, "X_keys.pkl"), "rb") as fp:
        X_keys = pickle.load(fp)
    # load morf features separatly
    morf = pd.DataFrame.from_dict(
        np.load(pathMorf, allow_pickle=True), orient="columns"
    )
    morf.drop(columns=['latitude', 'longitude'], inplace=True)
    X = pd.concat([X, morf], axis=1)
    # make specific order of features
    X = X[X_keys]
    X.replace([-float('inf'), float('inf'), np.nan, -3.3999999521443642e+38], 0,inplace=True)
    # Define scaler based on whole dataset
    scaler = joblib.load(os.path.join(path_to_npys_data, "scaler.save"))
    # Normalization using minmax scaler
    X = pd.DataFrame(scaler.transform(X), columns=X_keys)

    # Making predictions
    probabilities = make_predictions(X, X_keys, clf_dict)

    # Saving results:
    file_name = path.split("/")[-1]  # get the file name from the path
    ssp = file_name.split("_")[1]  # get sspXXX from the file name
    geo_model = file_name.split("_")[2].split(".")[0]  # get MRI/CNRM/CMCC from the file name

    for model in probabilities:
        with open(
            os.path.join(pathResults, "_".join([model, ssp, geo_model, "prob.npy"])),
            "wb",
        ) as f:
            pickle.dump(probabilities[model], f, protocol=4)


## Average probability based on different climate models

# cmcc = np.load(
#     os.path.join("..", "data", "results", "2040_2050", "lstm_ssp245" + "_CMCC_prob.npy"), allow_pickle=True
# )
# cnrm = np.load(
#     os.path.join("..", "data", "results", "2040_2050", "lstm_ssp245" + "_CNRM_prob.npy"), allow_pickle=True
# )
# mri = np.load(
#     os.path.join("..",  "data", "results", "2040_2050", "lstm_ssp245" + "_MRI_prob.npy"), allow_pickle=True
# )

# average = np.mean([cmcc, cnrm, mri], axis=0)
# with open(
#     os.path.join("..",  "data", "results", "2040_2050", "lstm_ssp245" + "_average_prob.npy"),
#     "wb",
# ) as f:
#     pickle.dump(average, f, protocol=4)
