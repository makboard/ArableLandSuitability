import os
import pickle
import sys

sys.path.append(os.path.join(".."))

import numpy as np
import pytorch_lightning as pl
import torch
import pandas as pd
from src.model_utils import (
    reshape_data,
    CropLSTM,
    CropMLP,
    CropConvLSTM,
    CropTransformer,
    CropPL,
    custom_multiclass_report,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple
from models.temperature_scaling import ModelWithTemperature
from catboost import CatBoostClassifier

def load_data(
    data_path: str, x_filename: str, y_filename: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads the data from specified files."""
    with open(os.path.join(data_path, x_filename), "rb") as fp:
        X = pickle.load(fp)

    with open(os.path.join(data_path, y_filename), "rb") as fp:
        y = pickle.load(fp)

    return X, y


def load_data_from_files() -> Dict[str, Any]:
    """
    Loads the data from the specified files.

    Returns:
        Dict[str, Any]: A dictionary with variable names as keys and their respective loaded data as values.
    """
    file_paths = [
        ("X_down_lstm.pkl", "X_lstm"),
        ("X_down.pkl", "X"),
        ("y_down.pkl", "y"),
        ("keys.pkl", "keys"),
    ]
    data = {}

    for file_name, var_name in file_paths:
        with open(
            os.path.join("..", "data", "processed_files", "pkls", file_name), "rb"
        ) as fp:
            data[var_name] = pickle.load(fp)

    return data


def evaluate_models_on_test_data(
    models_dict: Dict[str, str],
    X_test: np.ndarray,
    X_test_t: torch.Tensor,
    X_lstm: torch.Tensor,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Evaluates models on the test data and prints the custom multiclass report.

    Args:
        models_dict (Dict[str, str]): A dictionary with model names as keys and paths to their serialized data as values.
        X_test (np.ndarray): Test data.
        X_test_t (torch.Tensor): Tensor representation of test data.
        X_lstm (torch.Tensor): LSTM processed data for testing.
        y_test (np.ndarray): Ground truth labels for test data.

    Returns:
        Dict[str, np.ndarray]: A dictionary with model names as keys and their predicted probabilities as values.
    """
    y_probs = dict()
    softmax = nn.Softmax(dim=1)
    trainer = pl.Trainer(accelerator="gpu", devices=[2])

    for model in tqdm(models_dict):
        if model in ["mlp", "lstm", "transformer", "conv_lstm"]:
            if model == "mlp":
                network = CropMLP()
            elif model == "lstm":
                network = CropLSTM()
            elif model == "transformer":
                network = CropTransformer()
            elif model == "conv_lstm":
                network = CropConvLSTM()

            checkpoint = torch.load(models_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()

            # Use appropriate input tensor for the model
            input_tensor = (
                X_lstm if model in ["lstm", "transformer", "conv_lstm"] else X_test_t
            )
            predictions = torch.cat(
                trainer.predict(
                    loaded_model, DataLoader(input_tensor, batch_size=2048)
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_pred = np.argmax(y_prob, 1)
            y_probs[model] = y_prob
        elif model == "catboost":
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(models_dict[model])
            y_pred = loaded_model.predict(X_test)
            y_prob = loaded_model.predict_proba(X_test)
            y_probs[model] = y_prob
        else:
            loaded_model = pickle.load(open(models_dict[model], "rb"))
            y_pred = loaded_model.predict(X_test)
            y_prob = loaded_model.predict_proba(X_test)
            y_probs[model] = y_prob

        print(f"{model} results:")
        custom_multiclass_report(y_test, y_pred, y_probs[model])

    return y_probs


def make_predictions(
    models_dict: Dict[str, str],
    test_loader,
    test_loader_lstm,
    X: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Makes predictions on future data.

    Args:
        models_dict (Dict[str, str]): A dictionary with model names as keys and paths to their serialized data as values.
        test_loader (torch.Dataloader): Test dataloader for temperature scaling.
        test_loader_lstm (torch.Dataloader): Test dataloader in lstm format for temperature scaling.
        X (pd.DataFrame): future data.


    Returns:
        Dict[str, np.ndarray]: A dictionary with model names as keys and their predicted probabilities as values.
    """
    y_probs = dict()
    softmax = nn.Softmax(dim=1)
    temps = torch.linspace(0.25, 4.0, 100)
    trainer = pl.Trainer(accelerator="gpu", devices=[0])

    for model in tqdm(models_dict):
        if model in ["mlp", "lstm", "transformer", "conv_lstm"]:
            if model == "mlp":
                network = CropMLP()
            elif model == "lstm":
                network = CropLSTM()
            elif model == "transformer":
                network = CropTransformer()
            elif model == "conv_lstm":
                network = CropConvLSTM()

            checkpoint = torch.load(models_dict[model])
            loaded_model = CropPL(net=network)
            loaded_model.load_state_dict(checkpoint["state_dict"])
            loaded_model.eval()
            scaled_model = ModelWithTemperature(
                model=loaded_model.net,
                strategy="learn",
                per_class=True,
                device="cuda:0",
                verbose=True,
            )
            # Use appropriate input tensor for the model
            loader = (
                test_loader_lstm
                if model in ["lstm", "transformer", "conv_lstm"]
                else test_loader
            )
            scaled_model.set_temperature(loader, t_vals=list(temps))
            scaled_model = CropPL(net=scaled_model)

            # Use appropriate input tensor for the model
            input_tensor = (
                reshape_data(X) if model in ["lstm", "transformer", "conv_lstm"] else X
            )
            predictions = torch.cat(
                trainer.predict(
                    scaled_model, DataLoader(input_tensor, batch_size=2048)
                ),
                dim=0,
            )
            y_prob = softmax(predictions.float()).numpy()
            y_probs[model] = y_prob
        else:
            loaded_model = pickle.load(open(models_dict[model], "rb"))
            y_prob = loaded_model.predict_proba(X)
            y_probs[model] = y_prob

        print(f"{model} results:")

    return y_probs
