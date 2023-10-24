import copy
import os
import sys

sys.path.append("..")

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import xarray as xr
from src.dataprocessing import get_class_distribution
from descartes import PolygonPatch
from rasterio.crs import CRS
import matplotlib
from typing import Optional


def reproject_raster_to_match(
    data: xr.DataArray, match_data: xr.DataArray, crs_str: str = "EPSG:4326"
) -> xr.DataArray:
    """
    Reproject the data to match the resolution, projection, and region of another dataset.
    """
    data.rio.write_crs(crs_str, inplace=True)
    reprojected_data = data.rio.reproject_match(match_data)
    return xr.where(reprojected_data == 255, np.nan, reprojected_data)

def reproject_and_Feature_match(
    path_to_initial_target: str,
    path_to_any_feature: str,
    path_to_save: str,
    crs_str: str = "EPSG:4326",
    plot: Optional[bool] = False,
) -> None:
    """
    Reproject, resample and align to the spatial resolution of an example .tif file.

    Args:
        path_to_initial_target: Path to the original target variable .tif file.
        path_to_any_feature: Path to the example .tif file.
        path_to_save: Path to save the processed .tif file.
        crs_str: CRS string to define the coordinate reference system.
        plot: If True, plot resulting dataset.
    """
    
    # Open data
    Target = rxr.open_rasterio(path_to_initial_target).squeeze()
    Feature = rxr.open_rasterio(path_to_any_feature).squeeze()

    # Reproject the target to match feature's crs
    reprojected_target = reproject_raster_to_match(Target, Feature, crs_str)

    # Display class distribution
    print("Initial class distribution:")
    get_class_distribution(Target.data)
    print("\nDistribution of classes after reprojection:")
    get_class_distribution(reprojected_target.data)
    print("\n")

    # Save reprojected target data
    reprojected_target.rio.to_raster(path_to_save)
    print("Reprojected raster saved in the intended folder.")

    # Optionally plot reprojected target data
    if plot:
        f, ax = plt.subplots(figsize=(20, 4))
        reprojected_target.plot.imshow(ax=ax)
        plt.show()


def reproject_and_Feature_match_extended(
    path_to_initial_target: str,
    path_to_any_feature: str,
    path_to_landcover_data: str,
    path_to_save: str,
    crs_str="EPSG:4326",
    plot: Optional[bool] = False,
) -> None:
    """
    Reproject, resample and align to spatial resolution of the example .tif file (feature, for instance).
    Extended means that landcover data is used for assigning additional classes

    Args:
        path_to_initial_target (str): path to original target variable .tif file
        path_to_any_feature (str): path to example .tif file (feature)
        path_to_save (str): path to save processed .tif file
        path_to_landcover_data (str): path to landcover .tif file
        crs_str (str, optional): str to define crs
        plot (bool, optional): If True, plot resulting dataset. Defaults to False.
    """

    # Open data
    Target = rxr.open_rasterio(path_to_initial_target).squeeze()  # type: ignore
    Target_new = copy.deepcopy(
        rxr.open_rasterio(path_to_any_feature)[0, :, :].squeeze()  # type: ignore
    )
    Landcover = rxr.open_rasterio(path_to_landcover_data).squeeze()  # type: ignore

    # Reassign classes in Landcover data
    Landcover = xr.where(
        Landcover == 1, 6, Landcover
    )  # Barren: at least of area 60% is non-vegetated barren (sand, rock, soil) or permanent snow/ice with less than 10% vegetation.
    Landcover = xr.where(
        Landcover == 2, 7, Landcover
    )  # Permanent Snow and Ice: at least 60% of area is covered by snow and ice for at least 10 months of the year.
    Landcover = xr.where(
        Landcover == 3, 8, Landcover
    )  # Water Bodies: at least 60% of area is covered by permanent water bodies.
    Landcover = xr.where(
        Landcover == 10, 9, Landcover
    )  # Dense Forests: tree cover >60% (canopy >2m).
    Landcover = xr.where(
        Landcover == 20, 10, Landcover
    )  # Open Forests: tree cover 10-60% (canopy >2m).
    Landcover = xr.where(
        Landcover == 27, 11, Landcover
    )  # Woody Wetlands: shrub and tree cover >10% (>1m). Permanently or seasonally inundated.
    Landcover = xr.where(
        Landcover == 30, 12, Landcover
    )  # Grasslands: dominated by herbaceous annuals (<2m) >10% cover.
    Landcover = xr.where(
        Landcover == 40, 13, Landcover
    )  # Shrublands: shrub cover >60% (1-2m).
    Landcover = xr.where(
        Landcover == 50, 14, Landcover
    )  # Herbaceous Wetlands: dominated by herbaceous annuals (<2m) >10% cover. Permanently or seasonally inundated.
    Landcover = xr.where(
        Landcover == 51, 15, Landcover
    )  # Tundra: tree cover <10%. Snow-covered for at least 8 months of the year.

    # Reproject the data to another crs
    # Create a rasterio crs object for wgs 84 crs - lat / lon
    crs_wgs84 = CRS.from_string(crs_str)
    # Reproject the data using the crs object
    Target_wgs84 = Target.rio.reproject(crs_wgs84)
    Landcover.rio.write_crs("epsg:4326", inplace=True)
    Landcover = Landcover.rio.reproject(crs_wgs84)

    # Checking classes balance after reprojection
    print("Initial class distribution:")
    get_class_distribution(Target_wgs84.data)
    print("\n")

    # Apply reprojection based on Feature dimensions (Target_new)
    Target_new.rio.write_crs(crs_wgs84, inplace=True)
    Target_new = Target_wgs84.rio.reproject_match(Target_new)
    Landcover = Landcover.rio.reproject_match(Target_new)

    # Using Landcover data to replace 0 class at initial Target
    Target_new = xr.where(Target_new == 0, Landcover, Target_new)
    # Target_new = xr.where(Target_new == 255, Landcover_new, Target_new) # - optional
    Target_new = xr.where(Target_new == 255, np.nan, Target_new)

    # Checking classes balance after reprojection
    print("Distribution of classes after reprojection:")
    get_class_distribution(Target_new.data)
    print("\n")

    # saving target data
    Target_new.rio.to_raster(path_to_save)
    print("Reprojected raster saved in intended folder.")

    # Plot class distribution:
    if plot:
        # New target
        f, ax = plt.subplots(figsize=(20, 4))
        Target_new.plot.imshow(ax=ax)
        plt.show()


def pop_reproject_and_Feature_match(
    path_to_initial_target: str,
    path_to_any_feature: str,
    path_to_boundary: str,
    distribution_scenario: int,
    path_to_save: str,
    crs_str="EPSG:4326",
    plot: Optional[bool] = False,
) -> None:
    # Open data
    Target = rxr.open_rasterio(path_to_initial_target)[2].squeeze()  # type: ignore
    Target_new = copy.deepcopy(
        rxr.open_rasterio(path_to_any_feature)[0, :, :].squeeze()  # type: ignore
    )
    with fiona.open(path_to_boundary, "r") as sf:
        shapes = [feature["geometry"] for feature in sf]  # type: ignore
    patches = [
        PolygonPatch(shape, edgecolor="black", facecolor="none", linewidth=0.5)
        for shape in shapes
    ]

    # Reproject the data to another crs
    # Create a rasterio crs object for wgs 84 crs - lat / lon
    crs_wgs84 = CRS.from_string(crs_str)
    # Reproject the data using the crs object
    Target_wgs84 = Target.rio.reproject(crs_wgs84)

    # Repace Nan values with Zeros and log all values to decrease order of magnitude
    Target_wgs84 = Target_wgs84.fillna(np.nan)
    Target_wgs84 = np.log(Target_wgs84)

    # Then replace all -inf values with nans
    Target_wgs84 = Target_wgs84.where(np.isfinite).fillna(np.nan)

    # Apply reprojection based on Feature dimensions (Target_new)
    Target_new.rio.write_crs(crs_wgs84, inplace=True)
    Target_new = Target_wgs84.rio.reproject_match(
        Target_new
    )  # .interp_like(Target_new, method='nearest') #---- interp_like function yeilds slightly different results

    # Create an empty xarray
    pop_target_final = Target_new * 0

    if distribution_scenario == 1:
        # Distribution №1
        # Assign classes based on value range

        class0 = xr.where(np.isnan(Target_new), 0, 1) * 0
        class1 = xr.where((Target_new < -4.4) & (-15 < Target_new), 1, 0)
        class2 = xr.where((Target_new < -2.7) & (-4.4 < Target_new), 2, 0)
        class3 = xr.where((Target_new < -1.5) & (-2.7 < Target_new), 3, 0)
        class4 = xr.where((Target_new < -0.8) & (-1.5 < Target_new), 4, 0)
        class5 = xr.where((Target_new < 0.2) & (-0.8 < Target_new), 5, 0)
        class6 = xr.where((Target_new < 1.1) & (0.2 < Target_new), 6, 0)
        class7 = xr.where((Target_new < 2) & (1.1 < Target_new), 7, 0)
        class8 = xr.where((Target_new < 2.8) & (2 < Target_new), 8, 0)
        class9 = xr.where((Target_new < 3.8) & (2.8 < Target_new), 9, 0)
        class10 = xr.where((Target_new > 3.8), 10, 0)

        pop_target_final = (
            class0
            + class1
            + class2
            + class3
            + class4
            + class5
            + class6
            + class7
            + class8
            + class9
            + class10
        )
    elif distribution_scenario == 2:
        # Distribution №2
        # Assign classes based on value range

        class0 = xr.where(Target_new < -15, 0, 1) * 0
        class1 = xr.where((Target_new < -4.1) & (-15 < Target_new), 1, 0)
        class2 = xr.where((Target_new < -3) & (-4.4 < Target_new), 2, 0)
        class3 = xr.where((Target_new < -2.1) & (-3 < Target_new), 3, 0)
        class4 = xr.where((Target_new < -1.5) & (-2.1 < Target_new), 4, 0)
        class5 = xr.where((Target_new < -0.9) & (-1.5 < Target_new), 5, 0)
        class6 = xr.where((Target_new < -0.4) & (-0.9 < Target_new), 6, 0)
        class7 = xr.where((Target_new < 0.2) & (-0.4 < Target_new), 7, 0)
        class8 = xr.where((Target_new < 0.8) & (0.2 < Target_new), 8, 0)
        class9 = xr.where((Target_new < 1.3) & (0.8 < Target_new), 9, 0)
        class10 = xr.where((Target_new < 1.9) & (1.3 < Target_new), 10, 0)
        class11 = xr.where((Target_new < 2.4) & (1.9 < Target_new), 11, 0)
        class12 = xr.where((Target_new < 2.9) & (2.4 < Target_new), 12, 0)
        class13 = xr.where((Target_new < 3.5) & (2.9 < Target_new), 13, 0)
        class14 = xr.where((Target_new < 4.4) & (3.5 < Target_new), 14, 0)
        class15 = xr.where((Target_new > 4.4), 15, 0)

        pop_target_final = (
            class0
            + class1
            + class2
            + class3
            + class4
            + class5
            + class6
            + class7
            + class8
            + class9
            + class10
            + class11
            + class12
            + class13
            + class14
            + class15
        )

    # Checking classes balance after reprojection
    print("Distribution of classes after reprojection:")
    get_class_distribution(pop_target_final.data)
    print("\n")

    # saving target data
    pop_target_final.rio.to_raster(path_to_save)
    print("Reprojected raster saved in intended folder.")

    # Plot class distribution:
    if plot:
        # New target
        f, ax = plt.subplots(figsize=(20, 4))
        pop_target_final.plot.pcolormesh(
            ax=ax, cmap="terrain_r", add_colorbar=True, infer_intervals=True
        )
        ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))  # type: ignore
