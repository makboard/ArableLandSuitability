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

sys.path.append(os.path.join("..", ".."))

from src.model_utils import Crop_LSTM, Crop_PL, reshape_data


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
path_to_pickled_models = os.path.join("..", "results", "pickle_models")

clf_dict = {
    "lr": os.path.join(path_to_pickled_models, "Logistic_Regression_crops_final.pkl"),
    "xgbt": os.path.join(path_to_pickled_models, "XGBoost_crops_final.pkl"),
    "lgbm": os.path.join(path_to_pickled_models, "LightGBM_crops_final.pkl"),
    "MLP": os.path.join(path_to_pickled_models, "Crop_MLP.ckpt"),
    "lstm": os.path.join(path_to_pickled_models, "Crop_LSTM.ckpt"),
}


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
path_to_npys_data = os.path.join("..", "data", "npys_data")

pathFeatures = os.path.join(path_to_npys_data, "2040_2050", "features_ssp245_CNRM.npy")
pathFeaturesBaseline = os.path.join(path_to_npys_data, "features_initial_data_v1.npy")
pathResults = os.path.join("..", "results", "2040_2050")
pathMorf = os.path.join(path_to_npys_data, "features_morf_data.npy")

# %%
# Features
X = pd.DataFrame.from_dict(np.load(pathFeatures, allow_pickle=True), orient="columns")
X_baseline = pd.DataFrame.from_dict(
    np.load(pathFeaturesBaseline, allow_pickle=True), orient="columns"
)
with open(os.path.join(path_to_npys_data, "keys.pkl"), "rb") as fp:
    keys = pickle.load(fp)
morf = pd.DataFrame.from_dict(np.load(pathMorf, allow_pickle=True), orient="columns")
X = pd.concat([X, morf], axis=1)
X_baseline = pd.concat([X_baseline, morf], axis=1)
X = X[keys]
X = X.replace(-np.inf, 0)
X_baseline = X_baseline[keys]
X_baseline = X_baseline.replace(-np.inf, 0)

# Scaler used for training
scaler = joblib.load(os.path.join(path_to_npys_data, "scaler.save"))
cols = X.columns
X = reshape_data(pd.DataFrame(scaler.transform(X), columns=cols))
X_baseline = reshape_data(pd.DataFrame(scaler.transform(X_baseline), columns=cols))


# %%
def get_local_importances(
    X, X_baseline, boxes_list, names_list, target_list, device="cpu"
):
    with open(os.path.join(path_to_npys_data, "keys_lstm.pkl"), "rb") as fp:
        keys_lstm = pickle.load(fp)
    network = Crop_LSTM()
    checkpoint = torch.load(clf_dict["lstm"])
    loaded_model = Crop_PL(net=network)
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
            batch_size=128,
        )
        loader_baseline = torch.utils.data.DataLoader(
            torch.tensor(
                np.array(X_baseline_region),
                dtype=list(net.parameters())[0].dtype,
                device=device,
            ),
            batch_size=128,
        )
        heights_list = []
        kek = 0
        for input, baseline in tqdm(zip(loader_future, loader_baseline)):
            attribution_ig = ig.attribute(input, baselines=baseline, target=tar)
            heights_list.append(attribution_ig)
            

        bars = keys_lstm
        heights = torch.cat(heights_list).cpu().mean(dim=0).mean(dim=0).detach().numpy()
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


def plot_local_importances(order, fis):
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
    plt.savefig("../../results/feature_importance/lstm/local_importance.png")


# %%
china_bl = (121, 42.5)
china_tr = (128, 45)
ee_bl = (21, 45)
ee_tr = (45, 53)
ru_bl = (61, 57.5)
ru_tr = (80, 63)
order, fis = get_local_importances(
    X,
    X_baseline,
    [(ru_bl, ru_tr), (china_bl, china_tr), (ee_bl, ee_tr)],
    ["N-Russia", "NE-China", "E-EU"],
    target_list=[3, 1, 2],
    device=3,
)

# %%
plot_local_importances(order, fis)
