import xarray as xr
import numpy as np
import rioxarray as rxr
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
import statistics as st


def find_mode(window, axis=None, **kwargs):
    # find the mode over all axes
    uniq = np.unique(window, axis=0, return_counts=True)

    ret = uniq[0][:, np.argmax(uniq[1]), ...]
    uniq = np.unique(ret, axis=-1, return_counts=True)
    ret = uniq[0][..., np.argmax(uniq[1])]
    # ret = np.atleast_2d(ret)
    # retmax = np.max(window, axis, **kwargs)
    return ret


def get_cmip(
    path_to_cmip="/app/ArableLandSuitability/data/LUMIP_fracLut/fracLut_Emon_EC-Earth3-Veg_land-noLu_r1i1p1f1_gr_201001-201012.nc",
):
    arr = xr.open_dataset(path_to_cmip)
    return arr


def get_target(
    path_to_target="/app/ArableLandSuitability/data/target/target_croplands.tif",
):
    rds = rxr.open_rasterio(path_to_target)
    return rds


def cmp(path_to_target, path_to_cmip):
    pass


if __name__ == "__main__":
    from_cmip_to_target = False
    cmip = get_cmip()
    print("Order of variables in cmip:")
    print(cmip.sector.data)
    print("Crop is on 2 dim")
    cmip = cmip.drop_dims("bnds")
    cmip = cmip.mean(dim="time")
    cmip_crops = cmip.fracLut[2, ...]

    # interp cmip, avg over time
    target = get_target()
    target = target.rename({"y": "lat", "x": "lon"})
    target = target.squeeze()
    target.data = (target.data > 0).astype(int)
    lon_min, lon_max = target.lon.min().data, target.lon.max().data
    lat_min, lat_max = target.lat.min().data, target.lat.max().data
    if from_cmip_to_target:
        cmip_crops = cmip_crops.interp(
            lon=target.lon.data, lat=target.lat.data, method="linear"
        )
        cmip_crops = cmip_crops.reindex(lat=list(reversed(cmip_crops.lat)))
    else:
        cmip_crops = cmip_crops.sel(
            lat=slice(lat_min - 1, lat_max + 1), lon=slice(lon_min - 1, lon_max + 1)
        )
        cmip_crops = cmip_crops.reindex(lat=list(reversed(cmip_crops.lat)))
        lat_ratio = target.lat.data.shape[0] / cmip_crops.lat.data.shape[0]
        lon_ratio = target.lon.data.shape[0] / cmip_crops.lon.data.shape[0]
        target = (
            target.coarsen(lat=int(lat_ratio), lon=int(lon_ratio), boundary="trim")
            .max()
            .astype(int)
        )
        # target = target.interp(
        #     lon=cmip_crops.lon.data, lat=cmip_crops.lat.data, method="linear"
        # )
        pass

    tgt = target.data.flatten()
    preds = cmip_crops.data.flatten()
    precision, recall, thresholds = precision_recall_curve(tgt, preds)
    hmeans = 2 / (1 / recall[:-1] + 1 / precision[:-1])
    th = thresholds[np.argmax(hmeans)]
    print("ROCAUC", roc_auc_score(tgt, preds))
    print("AP", average_precision_score(tgt, preds))
    print("Accuracy", accuracy_score(tgt, preds > th))
    print("Precision", precision_score(tgt, preds > th))
    print("Recall", recall_score(tgt, preds > th))
    print("f1", f1_score(tgt, preds > th))

    pass
