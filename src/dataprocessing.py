import glob
import os
import pickle
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm


def get_file_paths(path_to_data: str, feature_names: list):
    """
    Filters out required features amongs terraclim dataset

    Arguments:
      path_to_data (str): path to directory that containts terraclim dataset
      feature_names (list): list of required features

    Returns:
      dict: key -- feature name; value -- list of related tif files
    """
    files_to_mosaic = glob.glob(path_to_data)
    files_to_mosaic = list(
        filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic)
    )
    file_paths = {
        fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names
    }
    return file_paths


def dataset_to_np(
    dataset: gdal.Dataset,
    x_off: int,
    y_off: int,
    xsize: int,
    ysize: int,
    verbose: bool = False,
):
    """
    Converts gdal.Dataset to numpy array
    !NB: raster bands are enumerated starting from 1!
    Arguments:
      dataset (gdal.Dataset): dataset to cast
      x_off (int): starting x position - idx
      y_off (int): starting y position - idx
      xsize (int): number of points to save in x direction
      ysize (int): number of points to save in y direction
    Returns:
      np.ndarray -- 3d tensor of information given in dataset
    """

    shape = [dataset.RasterCount, ysize, xsize]
    output = np.empty(shape)
    if verbose:
        for r_idx in range(shape[0]):
            band = dataset.GetRasterBand(r_idx + 1)
            arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
            output[r_idx, :, :] = np.array(arr)
    else:
        for r_idx in range(shape[0]):
            band = dataset.GetRasterBand(r_idx + 1)
            arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
            output[r_idx, :, :] = np.array(arr)
    return output


def get_nps_(file_path, verbose=False):  # IGRAN PROJECT
    # open gdal files
    dsets = {}
    for i in file_path:
        dset = gdal.Open(i)
        dsets[Path(i).stem] = dset
    # reading into np, scaling in accordance with terraclim provided
    nps = {}
    for fn in tqdm(dsets.keys()):
        np_tmp = dataset_to_np(
            dsets[fn],
            x_off=0,
            y_off=0,
            xsize=dsets[fn].RasterXSize,
            ysize=dsets[fn].RasterYSize,
            verbose=verbose,
        )
        nps[fn] = np_tmp

    return nps


def get_coords_res(dataset: gdal.Dataset):
    """
    For given dataset returns position of top left corner and resolutions

    Arguments:
      dataset (osgeo.gdal.Dataset): gdal dataset

    Returns:
      dict: containts coordinates of top left corner and
         resolutions alog x and y axes
    """
    gt = dataset.GetGeoTransform()
    output = {}
    output["x"] = gt[0]
    output["y"] = gt[3]
    output["x_res"] = gt[1]
    output["y_res"] = gt[-1]
    return output


def extract_latitude_longtitute(path_to_tifs: str, feature_name: str):
    """
    Extract 1d arrays of longitutde and latitude of given .tif data
    Helps to build a mapping between raster spatial indices and coordinates on the earth

    Arguments:
      path_to_data (str): path to directory that containts terraclim dataset
      feature_names (str): feature name of the interest

    Returns:
      tuple: longtitude and latitude 1d arrays
    """
    file_paths = get_file_paths(path_to_tifs, [feature_name])
    dset_tmp = gdal.Open(file_paths[feature_name][0])
    coords_dict = get_coords_res(dset_tmp)
    Lat = np.zeros((dset_tmp.RasterYSize))
    for i in range(Lat.shape[0]):
        Lat[i] = coords_dict["y"] + i * coords_dict["y_res"]
    Lon = np.zeros((dset_tmp.RasterXSize))
    for i in range(Lon.shape[0]):
        Lon[i] = coords_dict["x"] + i * coords_dict["x_res"]
    return Lat, Lon


def get_unique_values(data):
    # get unique values and counts of each value
    unique, counts = np.unique(data, return_counts=True)
    unique = list((x for x in unique))
    # display unique values and counts side by side
    values = []
    for i in range(len(unique)):
        values.append(tuple([unique[i], counts[i]]))
    values.sort(key=lambda tup: tup[1])
    return values


def get_class_distribution(data):
    tmp = get_unique_values(data)
    total = np.array(list(dict(tmp).values())).sum()
    for k, v in dict(tmp).items():
        print(f"Class {k} : {round(v / total * 100, 2)} %")


def get_target_data(path_to_raw_data, path_to_processed_data):
    with open(path_to_processed_data, "wb") as f:
        # Create dict to save at
        Target_dict = dict()
        # Save raster to npy
        np_tmp = dataset_to_np(
            gdal.Open(path_to_raw_data),
            x_off=0,
            y_off=0,
            xsize=gdal.Open(path_to_raw_data).RasterXSize,
            ysize=gdal.Open(path_to_raw_data).RasterYSize,
        )
        Target_dict["Target"] = np_tmp
        # Reshape data
        for key in Target_dict.keys():
            Target_dict[str(key)] = np.stack(
                [v for v in list(Target_dict[key][0, :, :])]
            ).reshape(-1)

        pickle.dump(Target_dict, f, protocol=4)


def get_features_data(path_to_raw_data, path_to_processed_data_folder):
    for key in path_to_raw_data:
        # Define file paths:
        file_path = glob.glob(os.path.join(path_to_raw_data[key], "*.tif"))

        # Features averaged
        with open(
            os.path.join(path_to_processed_data_folder, "features_" + key + ".npy"),
            "wb",
        ) as f:
            Features_dict = dict()
            # Get initial data
            data = get_nps_(file_path)

            # Monthly data
            for key in data.keys():
                if key in [
                    "fy",
                    "pr",
                    "pr_p95",
                    "sfcWindmax",
                    "snw",
                    "t0ud",
                    "tas",
                    "tasmax",
                    "tasmin",
                    "Tstep6",
                    "tp",
                    "t2m",
                    "monT0ud",
                    "monTstep6",
                ]:
                    # define time period in months
                    for i in range(12):
                        Features_dict[str(key) + "_M" + str(i + 1)] = np.stack(
                            [v for v in list(data[key][i, :, :])]
                        ).reshape(-1)
                # Morphological data
                elif key in ["morf_3", "morf_11", "morf_33", "morf_47"]:
                    for i in range(1, 11):
                        Features_dict[str(key) + "_" + str(i)] = np.stack(
                            [v for v in list(data[key][i - 1, :, :])]
                        ).reshape(-1)
                # Relief and SPI data
                elif key in ["DEM_1km", "12m_SPI"]:
                    Features_dict[str(key)] = np.stack(
                        [v for v in list(data[key][0, :, :])]
                    ).reshape(-1)

            pickle.dump(Features_dict, f, protocol=4)


def check_dimensions(path_to_features, path_to_target):
    features = np.load(
        os.path.join(path_to_features),
        allow_pickle=True,
    )
    dim_features = features[next(iter(features))].shape[0]

    target = np.load(
        os.path.join(path_to_target),
        allow_pickle=True,
    )
    dim_target = target[next(iter(target))].shape[0]

    if dim_features != dim_target:
        print("Error: Dimensions incompatible.")
    else:
        print("Ok!")
