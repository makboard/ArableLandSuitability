# %%
import os
import pickle
import sys

sys.path.append(os.path.join("..", ".."))
from src.model_utils import (CropLSTM,
                            CropMLP,
                            CropConvLSTM,
                            CropTransformer,
                            CropPL)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

from tqdm import tqdm
from copy import deepcopy

# Constants
NUM_PERMUTATION = 10

sklearn_binary_metric = f1_score

# defining paths
path_to_npys_data = os.path.join("..", "..", "data", "npys_data")
path_to_pkls = os.path.join("..", "..", "data", "processed_files", "pkls")
pathResults = os.path.join("..", "..", "results", "feature_importance")
if not os.path.exists(pathResults):
    os.makedirs(pathResults)
path_to_pickled_models = os.path.join("..", "..", "results", "pickle_models")

clf_dict = {
    # xgboost": os.path.join(path_to_pickled_models, "XGBoost.pkl"),
    "catboost": os.path.join(path_to_pickled_models, "catboost.pkl"),
    # "lr": os.path.join(path_to_pickled_models, "logreg.pkl"),
    # "conv_lstm": os.path.join(path_to_pickled_models, "conv_lstm.ckpt"),
    # "transformer": os.path.join(path_to_pickled_models, "transformer.ckpt"),
    "mlp": os.path.join(path_to_pickled_models,  "MLP.ckpt"),
    "lstm": os.path.join(path_to_pickled_models, "LSTM.ckpt"),
}

with open(
    os.path.join(path_to_pkls, "X_down.pkl"), "rb",
) as fp:
    X_mlp = pickle.load(fp)
    X_mlp = X_mlp["Test"].astype("float32")

with open(
    os.path.join(path_to_pkls, "X_down_lstm.pkl"), "rb",
) as fp:
    X_lstm = pickle.load(fp)
    X_lstm = X_lstm["Test"].astype("float32")

with open(
    os.path.join(path_to_pkls, "y_down.pkl"), "rb",
) as fp:
    y = pickle.load(fp)
    y = y["Test"].astype("float32")

with open(os.path.join(path_to_pkls, "keys.pkl"), "rb") as fp:
    mlp_keys = pickle.load(fp)
with open(os.path.join(path_to_pkls, "keys_lstm.pkl"), "rb") as fp:
    lstm_keys = pickle.load(fp)

X_dict = {"mlp": X_mlp, "lstm": X_lstm}
feature_names_dict = {
    "mlp": np.array(mlp_keys),
    "lstm": np.array(lstm_keys),
}

# Define a function to calculate model precision_score
def estimate_metrics(y_probs, y, sklearn_binary_metric):
    """
    Calculates metric
    
    Args:
        y_probs (Dict[str, np.ndarray]): Dictionary with name of a model and predicted probabilities.
        y (np.ndarray): Ground truth labels for test data.
        sklearn_binary_metric: Sklearn metric.

    Returns:
        loaded_model: Prefitted model.
    """
    scores = {}
    for key in y_probs.keys():
        if (key == "lstm") or (key == "transformer") or (key == "conv_lstm"):
            scores[key] = sklearn_binary_metric(
                y.argmax(1), y_probs[key].argmax(1), average="macro"
            )
        else:
            scores[key] = sklearn_binary_metric(
                y.argmax(1), y_probs[key].argmax(1), average="macro"
            )
    return scores


def load_model(model_name, model_path):
    """
    Loads a model from the path.
    
    Args:
        model_name (str): A name of the model.
        model_path (str): path to model.

    Returns:
        loaded_model: Prefitted model.
    """
    if model_name in ["mlp", "lstm", "transformer", "conv_lstm"]:
        if model_name == "mlp":
            network = CropMLP()
        elif model_name == "lstm":
            network = CropLSTM()
        elif model_name == "transformer":
            network = CropTransformer()
        elif model_name == "conv_lstm":
            network = CropConvLSTM()
            
        checkpoint = torch.load(model_path)
        loaded_model = CropPL(net=network)
        loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()

    elif model_name=="catboost":
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(model_path)
    else:
            loaded_model = pickle.load(open(model_path, "rb"))
    
    return loaded_model


def predict_prob(model_name,
                loaded_model,
                X_test: np.ndarray,
                X_test_t: torch.Tensor,
                X_lstm: torch.Tensor) -> dict:
                # y_test: np.ndarray
    """
    Evaluates model on the test data.
    
    Args:
        model_name (str): A name of the model.
        loaded_model: Prefitted model.
        joint_dict (Dict[str, str]): A dictionary with model names as keys and paths to their serialized data as values.
        X_test (np.ndarray): Test data.
        X_test_t (torch.Tensor): Tensor representation of test data.
        X_lstm (torch.Tensor): LSTM processed data for testing.
        # y_test (np.ndarray): Ground truth labels for test data.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary with model names as keys and their predicted probabilities as values.
    """
    y_probs = dict()
    softmax = nn.Softmax(dim=1)
    trainer = pl.Trainer(accelerator="gpu", devices=[0])

    if model_name in ["mlp", "lstm", "transformer", "conv_lstm"]:

        # Use appropriate input tensor for the model
        input_tensor = X_lstm if model_name in ["lstm", "transformer", "conv_lstm"] else X_test_t
        predictions = torch.cat(
            trainer.predict(loaded_model, DataLoader(input_tensor, batch_size=2048)), dim=0
        )
        y_prob = softmax(predictions.float()).numpy()
        # y_pred = np.argmax(y_prob, 1)
        y_probs[model_name] = y_prob

    elif model_name =="catboost":
        y_prob = loaded_model.predict_proba(X_test)
        y_probs[model_name] = y_prob

    else:
        y_prob = loaded_model.predict_proba(X_test)
        y_probs[model_name] = y_prob

    return y_probs

def estimate_feature_importance_models(clf_dict,
                                        X_dict,
                                        y,
                                        feature_names_dict,
                                        num_permutations,
                                        sklearn_binary_metric):
    """
    Estimates the importance of the features and saves it to pickle.
    
    Args:
        clf_dict (Dict[str, str]): A dictionary with model names as keys and paths to their serialized data as values.
        X_dict (dict): Test data.
        y (np.ndarray): Ground truth labels for test data.
        feature_names_dict (Dict[str, list]): A dictionary with lists of features as values
        num_permutations (int): Number of permutations to run.
        sklearn_binary_metric (str): Metric to use.
    """

    # Calculate original score
    for model_name in tqdm(clf_dict):
        print("Loading model")
        loaded_model = load_model(model_name, clf_dict[model_name])
        print("Predicting on test data")
    
        y_probs = predict_prob (model_name,
                            loaded_model,
                            X_dict["mlp"],
                            torch.tensor(X_dict["mlp"]),
                            torch.tensor(X_dict["lstm"]))

        original_score = estimate_metrics(y_probs, y, sklearn_binary_metric)

        # Initialize an array to store the feature importance scores
        feature_importance_scores = {
            model_name: {} for model_name in y_probs.keys()
            }

        # Collect list of features
        if model_name=="lstm" or model_name=="conv_lstm" or model_name=="transformer":
            feature_names = feature_names_dict["lstm"]
        else:
            feature_names = feature_names_dict["mlp"]
        
        feature_importance_scores[model_name] = {fn: [] for fn in feature_names}

        # Loop over each feature
        for feature_dim, feature_name in tqdm(enumerate(feature_names)):
            print(f"Calculating importance for feature {feature_name}")

            # Loop over each permutation
            for i in range(num_permutations):
                # Create a copy of the input data for shuffling
                shuffled_input_data = deepcopy(X_dict)

                if model_name=="lstm" or model_name=="conv_lstm" or model_name=="transformer":
                    np.random.shuffle(shuffled_input_data["lstm"][:, :, feature_dim])

                else:
                    np.random.shuffle(shuffled_input_data["mlp"][:, feature_dim])

                # make prediction
                shuffled_preds = predict_prob(model_name, loaded_model,
                                            shuffled_input_data["mlp"],
                                            torch.tensor(shuffled_input_data["mlp"]),
                                            torch.tensor(shuffled_input_data["lstm"]))

                # for each model estimate chosen metric
                shuffled_score = estimate_metrics(
                    shuffled_preds, y, sklearn_binary_metric
                )

                # Calculate the difference in accuracy between the shuffled and original input data
                importance_score = original_score[model_name] - shuffled_score[model_name] 

                del shuffled_input_data

                # Add the importance score to the array
                for model_name in original_score.keys():
                    feature_importance_scores[model_name][feature_name].append(
                        importance_score
                    )
            
        # Save metrics
        filename = "feature_importance_{}.pkl".format(model_name)
        with open(
            os.path.join(pathResults, filename), "wb"
        ) as fs:
            pickle.dump(feature_importance_scores[model_name], fs)


def main():
    estimate_feature_importance_models(clf_dict,
                                        X_dict,
                                        y,
                                        feature_names_dict,
                                        NUM_PERMUTATION,
                                        sklearn_binary_metric,
    )

if __name__ == "__main__":
    main()