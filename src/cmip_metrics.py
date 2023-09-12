import xarray as xr
import rioxarray as rxr
from sklearn.metrics import classification_report

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
    cmip_crops_inter = cmip_crops.interp(
        lon=target.lon.data, lat=target.lat.data, method="nearest"
    )

    tgt = target.data.flatten()
    preds = cmip_crops_inter.data.flatten()
    print(classification_report(tgt, preds > 0.5))
    pass
