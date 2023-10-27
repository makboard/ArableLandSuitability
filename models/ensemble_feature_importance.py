# %%
import os
import pickle
import sys

sys.path.append(os.path.join("/app/ArableLandSuitability"))

import numpy as np
from torch import nn
from sklearn.metrics import f1_score

from src.predict import make_predictions
from src.model_utils import custom_multiclass_report
import src.feature_importance as fi


# defining paths
path_to_npys_data = os.path.join("/app/ArableLandSuitability", "data", "npys_data")

# pathFeatures = os.path.join(path_to_npys_data, "2040_2050")
pathResults = os.path.join("/app/ArableLandSuitability", "results", "ensemble")
if not os.path.exists(pathResults):
    os.makedirs(pathResults)
# pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

softmax = nn.Softmax()

path_to_pickled_models = os.path.join(
    "/app/ArableLandSuitability", "data", "results", "pickle_models"
)

clf_dict = {
    # "catboost": os.path.join(path_to_pickled_models, "catboost.pkl"),
    # "lr": os.path.join(path_to_pickled_models, "logreg.pkl"),
    "conv_lstm": os.path.join(path_to_pickled_models, "conv_lstm.pkl"),
    # "transformer": os.path.join(path_to_pickled_models, "transformer.pkl"),
    # "MLP": os.path.join(path_to_pickled_models,  "mlp.pkl"),
    # "lstm": os.path.join(path_to_pickled_models, "lstm.pkl"),
}

with open(
    os.path.join(
        "/app/ArableLandSuitability", "data", "processed_files", "pkls", "X.pkl"
    ),
    "rb",
) as fp:
    X_mlp = pickle.load(fp)
    X_mlp = X_mlp["Test"].astype("float32")

with open(
    os.path.join(
        "/app/ArableLandSuitability", "data", "processed_files", "pkls", "X_lstm.pkl"
    ),
    "rb",
) as fp:
    X_lstm = pickle.load(fp)
    X_lstm = (X_lstm["Test"][0].astype("float32"), X_lstm["Test"][1].astype("float32"))

with open(
    os.path.join(
        "/app/ArableLandSuitability", "data", "processed_files", "pkls", "y.pkl"
    ),
    "rb",
) as fp:
    y_mlp = pickle.load(fp)
    y_mlp = y_mlp["Test"].astype("float32")
with open(
    os.path.join(
        "/app/ArableLandSuitability", "data", "processed_files", "pkls", "y_lstm.pkl"
    ),
    "rb",
) as fp:
    y_lstm = pickle.load(fp)
    y_lstm = y_lstm["Test"].astype("float32")

# with open(
#     os.path.join("/app/ArableLandSuitability", "data", "npys_data", "alpha.pkl"), "rb"
# ) as fp:
#     weight = pickle.load(fp)
with open(os.path.join(path_to_npys_data, "X_keys.pkl"), "rb") as fp:
    mlp_keys = pickle.load(fp)
with open(os.path.join(path_to_npys_data, "monthly_keys.pkl"), "rb") as fp:
    monthly_keys = pickle.load(fp)
with open(os.path.join(path_to_npys_data, "static_keys.pkl"), "rb") as fp:
    static_keys = pickle.load(fp)

X_dict = {"mlp": X_mlp, "lstm": X_lstm}
y_dict = {"mlp": y_mlp, "lstm": y_lstm}
feature_names_dict = {
    "mlp": np.array(mlp_keys),
    "monthly": np.array(monthly_keys).reshape((X_lstm[0].shape[-1], X_lstm[0].shape[-2])),
    "static": np.array(static_keys),
}

lstm_keys = set(monthly_keys) | set(static_keys)
assert sorted(mlp_keys) == sorted(
    lstm_keys
), "The sequential and non sequential models have different features"

# FIs
num_permutations = 2
sklearn_binary_metric = f1_score
fi.estimate_feature_importance_models(
    clf_dict,
    X_dict,
    y_dict,
    feature_names_dict,
    num_permutations,
    sklearn_binary_metric,
)
# pred_probas = make_predictions(X_dict, clf_dict)
# pred_probas["mean_ensemble_" + "_".join(list(pred_probas.keys()))] = sum(
#     list(pred_probas.values())
# ) / len(clf_dict)

# # custom_multiclass_report()
# for key in pred_probas.keys():
#     path_model_pics = os.path.join(pathResults, key)
#     if not os.path.exists(path_model_pics):
#         os.makedirs(path_model_pics)
#     if key == "lstm":
#         custom_multiclass_report(
#             y_dict["lstm"].argmax(1),
#             pred_probas[key].argmax(1),
#             pred_probas[key],
#             path_to_save=path_model_pics,
#         )
#     else:
#         custom_multiclass_report(
#             y_dict["mlp"].argmax(1),
#             pred_probas[key].argmax(1),
#             pred_probas[key],
#             path_to_save=path_model_pics,
#         )
# pass
#

# Saving results:
# file_name = path.split("/")[-1]  # get the file name from the path
# ssp = file_name.split("_")[1]  # get sspXXX from the file name
# geo_model = file_name.split("_")[2].split(".")[
#     0
# ]  # get MRI/CNRM/CMCC from the file name

# for model in pred_probas:
#     with open(
#         os.path.join(pathResults, "_".join([model, "prob.npy"])),
#         "wb",
#     ) as f:
#         pickle.dump(pred_probas[model], f, protocol=4)


# %% [markdown]
# ## Average probability based on different climate models

# %%
# cmcc = np.load(
#     os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_CMCC_prob.npy"),
#     allow_pickle=True,
# )
# cnrm = np.load(
#     os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_CNRM_prob.npy"),
#     allow_pickle=True,
# )
# mri = np.load(
#     os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_MRI_prob.npy"),
#     allow_pickle=True,
# )

# average = np.mean([cmcc, cnrm, mri], axis=0)
# with open(
#     os.path.join("..", "results", "2040_2050", "lstm_ssp245" + "_average_prob.npy"),
#     "wb",
# ) as f:
#     pickle.dump(average, f, protocol=4)
