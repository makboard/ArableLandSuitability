# %%
import os
import pickle
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
import torch
from captum.attr import IntegratedGradients
from shapely import geometry
from tqdm import tqdm

sys.path.append("/app/ArableLandSuitability")

from src.model_utils import reshape_data
from src.model_utils import CropMLP, CropTransformer, CropLSTM, CropConvLSTM, CropPL


def load_X_and_baseline(data_type, load_pre_saved):
    if not load_pre_saved:
        X = pd.DataFrame.from_dict(
            np.load(pathFeatures, allow_pickle=True), orient="columns"
        )
        X_baseline = pd.DataFrame.from_dict(
            np.load(pathFeaturesBaseline, allow_pickle=True), orient="columns"
        )
        morf = pd.DataFrame.from_dict(
            np.load(pathMorf, allow_pickle=True), orient="columns"
        )
        X = pd.concat([X, morf], axis=1)
        X_baseline = pd.concat([X_baseline, morf], axis=1)
        X = X[mlp_keys]
        X = X.replace(-np.inf, 0)
        X = X.fillna(0)
        X = X.replace(np.inf, 0)
        X_baseline = X_baseline[mlp_keys]
        X_baseline = X_baseline.replace(-np.inf, 0)
        X_baseline = X_baseline.fillna(0)
        X_baseline = X_baseline.replace(np.inf, 0)

        # Scaler used for training
        scaler = joblib.load(os.path.join(path_to_npys_data, "scaler.save"))
        cols = X.columns
        X = scaler.transform(X)
        X_baseline = scaler.transform(X_baseline)
        if data_type == "mlp":
            with open(
                "/app/ArableLandSuitability/data/npys_data/X_future_2040_2050_CNRM.npy",
                "wb",
            ) as fs:
                np.save(fs, X)
            with open(
                "/app/ArableLandSuitability/data/npys_data/X_baseline.npy", "wb"
            ) as fs:
                np.save(fs, X_baseline)
            return X, X_baseline
        elif data_type == "sequential":
            X_lstm = reshape_data(pd.DataFrame(X, columns=cols))
            X_baseline_lstm = reshape_data(pd.DataFrame(X_baseline, columns=cols))
            with open(
                "/app/ArableLandSuitability/data/npys_data/X_lstm_future_2040_2050_CNRM.npy",
                "wb",
            ) as fs:
                np.save(fs, X_lstm)
            with open(
                "/app/ArableLandSuitability/data/npys_data/X_baseline_lstm.npy", "wb"
            ) as fs:
                np.save(fs, X_baseline_lstm)
            return X_lstm, X_baseline_lstm
        else:
            raise NotImplementedError("type should be `mlp` or `sequential`")
    else:
        if data_type == "mlp":
            X = np.load(
                "/app/ArableLandSuitability/data/npys_data/X_future_2040_2050_CNRM.npy"
            )
            X_baseline = np.load(
                "/app/ArableLandSuitability/data/npys_data/X_baseline.npy"
            )
            return X, X_baseline
        elif data_type == "sequential":
            X_lstm = np.load(
                "/app/ArableLandSuitability/data/npys_data/X_lstm_future_2040_2050_CNRM.npy"
            )
            X_baseline_lstm = np.load(
                "/app/ArableLandSuitability/data/npys_data/X_baseline_lstm.npy"
            )
            return X_lstm, X_baseline_lstm
        else:
            raise NotImplementedError("type should be `mlp` or `sequential`")


# %%
def get_coord_idxs(lon_lat):
    xs = np.array(range(width))
    ys = np.array(range(height))
    lon_idx = np.argmin(np.abs(lon_lat[0] - (left + delta_lon * xs)))
    lat_idx = np.argmin(np.abs(lon_lat[1] - (bottom + delta_lat * ys)))
    return (lon_idx, lat_idx)


def get_square(arr, tl_idxs, br_idxs, reshape_to_map=False):
    if reshape_to_map:
        return np.concatenate(
            [
                arr[tl_idxs[0] + i * width : br_idxs[0] + i * width]
                for i in range(tl_idxs[1], br_idxs[1])
            ]
        ).reshape(br_idxs[1] - tl_idxs[1], -1)
    else:
        try:
            return pd.concat(
                [
                    arr[tl_idxs[0] + i * width : br_idxs[0] + i * width]
                    for i in range(tl_idxs[1], br_idxs[1])
                ]
            )
        except TypeError:
            return np.concatenate(
                [
                    arr[tl_idxs[0] + i * width : br_idxs[0] + i * width]
                    for i in range(tl_idxs[1], br_idxs[1])
                ]
            )


# %%
months = [f"{i:02d}" for i in range(1, 13)]
years = np.arange(2000, 2010)
years_future = np.arange(2020, 2030)


# %%
# path_to_pickled_models = os.path.join("..", "results", "pickle_models")

# clf_dict = {
#     "lr": os.path.join(path_to_pickled_models, "Logistic_Regression_crops_final.pkl"),
#     "xgbt": os.path.join(path_to_pickled_models, "XGBoost_crops_final.pkl"),
#     "lgbm": os.path.join(path_to_pickled_models, "LightGBM_crops_final.pkl"),
#     "MLP": os.path.join(path_to_pickled_models, "Crop_MLP.ckpt"),
#     "lstm": os.path.join(path_to_pickled_models, "Crop_LSTM.ckpt"),
# }


# %%
left = 20
top = 64
right = 152
bottom = 42

p1 = geometry.Point(left, bottom)
p2 = geometry.Point(left, top)
p3 = geometry.Point(right, top)
p4 = geometry.Point(right, bottom)

pointList = [p1, p2, p3, p4]
poly = geometry.Polygon([i for i in pointList])


# %%
# Define the parameters of transformation
width = 9044
height = 1508

bbox_size = (height, width)
bbox = [left, bottom, right, top]
transform = rasterio.transform.from_bounds(
    *bbox, width=bbox_size[1], height=bbox_size[0]
)


# %%
delta_lon = (right - left) / width
delta_lat = (top - bottom) / height

# %%
# defining paths
path_to_npys_data = os.path.join("data", "npys_data")

pathFeatures = os.path.join(path_to_npys_data, "2040_2050", "features_ssp245_CNRM.npy")
pathFeaturesBaseline = os.path.join(path_to_npys_data, "features_initial_data.npy")
pathResults = os.path.join("results", "2040_2050")
pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

# %%
# Features
data_type = "mlp"
if data_type == "mlp":
    X, X_baseline = load_X_and_baseline(data_type, True)
elif data_type == "sequential":
    X_lstm, X_baseline_lstm = load_X_and_baseline(data_type, True)
else:
    raise NotImplementedError("data_type should be either `mlp` or `sequential`")
with open(os.path.join(path_to_npys_data, "X_keys.pkl"), "rb") as fp:
    mlp_keys = pickle.load(fp)
with open(os.path.join(path_to_npys_data, "monthly_keys.pkl"), "rb") as fp:
    monthly_keys = pickle.load(fp)
with open(os.path.join(path_to_npys_data, "static_keys.pkl"), "rb") as fp:
    static_keys = pickle.load(fp)

# feature_names_dict = {
#     "mlp": np.array(mlp_keys),
#     "monthly": np.array(monthly_keys),
#     "static": np.array(static_keys),
# }
lstm_keys_ = ["_".join(v.split("_")[:-1]) for v in monthly_keys]
lstm_keys = []
for v in lstm_keys_:
    if v not in lstm_keys:
        lstm_keys.append(v)
for v in static_keys:
    lstm_keys.append(v)
# lstm_keys = set(monthly_keys) | set(static_keys)
# assert sorted(mlp_keys) == sorted(
#     lstm_keys
# ), "The sequential and non sequential models have different features"
mlp_keys.remove("latitude")
mlp_keys.remove("longitude")
lstm_keys.remove("latitude")
lstm_keys.remove("longitude")


# %%
def get_local_importances(
    path_to_ckpt,
    model,
    X,
    X_baseline,
    boxes_list,
    names_list,
    target_list,
    device="cpu",
):
    # loaded_model = torch.load(path_to_pkl)
    # loaded_model.eval()
    if model == "mlp":
        keys = mlp_keys
        network = CropMLP()
    elif model == "lstm":
        keys = lstm_keys
        network = CropLSTM()
    elif model == "transformer":
        keys = lstm_keys
        network = CropTransformer()
    elif model == "conv_lstm":
        keys = lstm_keys
        network = CropConvLSTM()
    loaded_model = CropPL(net=network)
    # network = Crop_LSTM()
    checkpoint = torch.load(path_to_ckpt)
    # loaded_model = Crop_PL(net=network)
    loaded_model.load_state_dict(checkpoint["state_dict"])
    net = loaded_model.net
    net.to(device=device)
    order = []
    fis = {}
    for box, name, tar in zip(boxes_list, names_list, target_list):
        print(name)
        bl_idxs = get_coord_idxs(box[0])
        tr_idxs = get_coord_idxs(box[1])
        X_region = get_square(X, bl_idxs, tr_idxs)
        X_baseline_region = get_square(X_baseline, bl_idxs, tr_idxs)
        ig = IntegratedGradients(net)
        loader_future = torch.utils.data.DataLoader(
            torch.tensor(
                np.array(X_region), dtype=list(net.parameters())[0].dtype, device=device
            ),
            batch_size=64,
        )
        loader_baseline = torch.utils.data.DataLoader(
            torch.tensor(
                np.array(X_baseline_region),
                dtype=list(net.parameters())[0].dtype,
                device=device,
            ),
            batch_size=64,
        )
        heights_list = []
        kek = 0
        for input, baseline in tqdm(zip(loader_future, loader_baseline)):
            attribution_ig = ig.attribute(input, baselines=baseline, target=tar)
            heights_list.append(attribution_ig)

        bars = keys
        if model == "mlp":
            heights = torch.cat(heights_list).cpu().mean(dim=0).detach().numpy()
        else:
            heights = (
                torch.cat(heights_list).cpu().mean(dim=0).mean(dim=0).detach().numpy()
            )
        d = {"bars": bars, name: heights}

        if len(order) == 0:
            df_plot = pd.DataFrame.from_dict(d).sort_values(name, key=lambda x: abs(x))[
                ::-1
            ]
            order = df_plot["bars"]
        else:
            d_ordered = {"bars": order}
            df_plot_non_ordered = pd.DataFrame.from_dict(d)
            heights_order = []
            for key in order:
                heights_order.append(
                    df_plot_non_ordered[df_plot_non_ordered["bars"] == key][
                        name
                    ].values[0]
                )
            d_ordered[name] = heights_order
            df_plot = pd.DataFrame.from_dict(d_ordered)
        fis[name] = df_plot[name].values[:10]
    order = order[:10]
    return order, fis


def plot_local_importances(model, order, fis):
    x = np.arange(len(order))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in fis.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Attribution")
    ax.set_xticks(x + width, order, rotation=90)
    ax.legend(loc="upper left", ncols=3)

    res_fi_path = "/app/ArableLandSuitability/results/feature_importance/"
    if not os.path.exists(os.path.join(res_fi_path, model)):
        os.makedirs(os.path.join(res_fi_path, model))
    with open(os.path.join(res_fi_path, model, "fis.npy"), "wb") as fs:
        np.save(fs, fis)
    with open(os.path.join(res_fi_path, model, "order.npy"), "wb") as fs:
        np.save(fs, order)
    plt.savefig(os.path.join(os.path.join(res_fi_path, model, "local_importance.png")))


# %%
china_bl = (121, 42.5)
china_tr = (128, 45)
ee_bl = (25, 52)
ee_tr = (38, 58)
ru_bl = (60, 55)
ru_tr = (70, 60)
clf_dict = {
    "mlp": "/app/ArableLandSuitability/results/pickle_models/MLP.ckpt",
    # "lstm": "/app/ArableLandSuitability/results/pickle_models/LSTM.ckpt",
    # "transformer": "/app/ArableLandSuitability/results/pickle_models/transformer.ckpt",
    # "conv_lstm": "/app/ArableLandSuitability/results/pickle_models/conv_lstm.ckpt",
}
if data_type == "mlp":
    assert (
        ("mlp" in clf_dict.keys())
        and ("lstm" not in clf_dict.keys())
        and ("transformer" not in clf_dict.keys())
        and ("conv_lstm" not in clf_dict.keys())
    ), "Attempted to run sequential models with mlp data"
elif data_type == "sequential":
    assert ("mlp" not in clf_dict.keys()) and (
        ("lstm" in clf_dict.keys())
        or ("transformer" in clf_dict.keys())
        or ("conv_lstm" in clf_dict.keys())
    ), "Attempted to run mlp models with sequential data"
else:
    raise NotImplementedError("type should be `mlp` or `sequential`")

for model_key, model_pkl_path in tqdm(clf_dict.items()):
    if model_key == "mlp":
        order, fis = get_local_importances(
            model_pkl_path,
            model_key,
            X,
            X_baseline,
            [(ru_bl, ru_tr), (china_bl, china_tr), (ee_bl, ee_tr)],
            ["N-Russia", "NE-China", "E-EU"],
            target_list=[3, 1, 2],
            device=1,
        )

        # %%
        plot_local_importances(model_key, order, fis)
    else:
        order, fis = get_local_importances(
            model_pkl_path,
            model_key,
            X_lstm,
            X_baseline_lstm,
            [(ru_bl, ru_tr), (china_bl, china_tr), (ee_bl, ee_tr)],
            ["N-Russia", "NE-China", "E-EU"],
            target_list=[3, 1, 2],
            device=1,
        )

        # %%
        plot_local_importances(model_key, order, fis)
