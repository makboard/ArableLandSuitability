import pytorch_lightning as pl
import pickle
import torch
from src.model_utils import (
    CropLSTM,
    CropMLP,
    CropPL,
    reshape_data,
    CroplandDatasetPredict,
)
from torch import nn
from torch.utils.data import DataLoader
from catboost import CatBoostClassifier
from tqdm import tqdm


def make_prediction_model(X_dict, model, path_to_pkl, batch_size):
    if model == "MLP":
        # network = CropMLP()
        loaded_model = torch.load(path_to_pkl)
        # loaded_model = CropPL(net=network)
        # loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()

        # create an instance of pl.Trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1)

        # check metrics
        predictions = torch.cat(
            trainer.predict(
                loaded_model,
                DataLoader(
                    torch.tensor(X_dict["mlp"], dtype=torch.float),
                    batch_size=batch_size,
                ),
            ),
            dim=0,
        )
        softmax = nn.Softmax(dim=1)
        y_prob = softmax(predictions.float()).numpy()

    elif model == "lstm":
        # network = CropLSTM()
        loaded_model = torch.load(path_to_pkl)
        # loaded_model = CropPL(net=network)
        # loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()

        # create an instance of pl.Trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1)

        # check metrics
        predictions = torch.cat(
            trainer.predict(
                loaded_model,
                DataLoader(
                    # torch.tensor(X_dict["lstm"], dtype=torch.float),
                    CroplandDatasetPredict(X_dict["lstm"]),
                    batch_size=batch_size,
                ),
            ),
            dim=0,
        )
        softmax = nn.Softmax(dim=1)
        y_prob = softmax(predictions.float()).numpy()

    elif model == "transformer":
        # network = CropLSTM()
        loaded_model = torch.load(path_to_pkl)
        # loaded_model = CropPL(net=network)
        # loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()

        # create an instance of pl.Trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1)

        # check metrics
        predictions = torch.cat(
            trainer.predict(
                loaded_model,
                DataLoader(
                    # torch.tensor(X_dict["lstm"], dtype=torch.float),
                    CroplandDatasetPredict(X_dict["lstm"]),
                    batch_size=batch_size,
                ),
            ),
            dim=0,
        )
        softmax = nn.Softmax(dim=1)
        y_prob = softmax(predictions.float()).numpy()

    elif model == "conv_lstm":
        # network = CropLSTM()
        loaded_model = torch.load(path_to_pkl)
        # loaded_model = CropPL(net=network)
        # loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()

        # create an instance of pl.Trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1)

        # check metrics
        predictions = torch.cat(
            trainer.predict(
                loaded_model,
                DataLoader(
                    # torch.tensor(X_dict["lstm"], dtype=torch.float),
                    CroplandDatasetPredict(X_dict["lstm"]),
                    batch_size=batch_size,
                ),
            ),
            dim=0,
        )
        softmax = nn.Softmax(dim=1)
        y_prob = softmax(predictions.float()).numpy()

    elif model == "catboost":
        # loading the models:
        print("Table ML:", model)
        # loaded_model = pickle.load(open(path_to_pkl, "rb"))
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(path_to_pkl)
        y_prob = loaded_model.predict_proba(X_dict["mlp"])

    elif model == "lr":
        # loading the models:
        print("Table ML:", model)
        loaded_model = pickle.load(open(path_to_pkl, "rb"))
        y_prob = loaded_model.predict_proba(X_dict["mlp"])

    else:
        raise NotImplementedError("model not recognized")
    return y_prob


def make_predictions(X_dict, clf_dict, batch_size=8192):
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
        y_probs[model] = make_prediction_model(
            X_dict=X_dict,
            model=model,
            path_to_pkl=clf_dict[model],
            batch_size=batch_size,
        )

    return y_probs
