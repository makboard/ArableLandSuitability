import glob
import os
import pickle
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm
from typing import List, Dict, Union, Tuple, Iterable


def get_file_paths(path_to_data: str, feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Filters out required features among the TerraClim dataset.

    Parameters:
    -----------
    path_to_data : str
        Path to the directory that contains the TerraClim dataset.
    feature_names : List[str]
        List of required feature names.

    Returns:
    --------
    Dict[str, List[str]]:
        Dictionary where the key is the feature name and the value is a list of related .tif files.
    """
    
    all_files = glob.glob(f"{path_to_data}/*.tif")  # Assuming files have .tif extension.
    
    # Filter the files based on feature names
    files_to_mosaic = [file for file in all_files if any(fn in file for fn in feature_names)]
    
    # Organize the files into a dictionary based on feature names
    file_paths = {fn: [file for file in files_to_mosaic if fn in file] for fn in feature_names}
    
    return file_paths


def dataset_to_np(
    dataset: gdal.Dataset,
    x_off: int,
    y_off: int,
    xsize: int,
    ysize: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Converts gdal.Dataset to numpy array.
    
    Note: Raster bands in gdal are enumerated starting from 1.
    
    Parameters:
    -----------
    dataset : gdal.Dataset
        Dataset to cast to numpy array.
    x_off : int
        Starting x position index.
    y_off : int
        Starting y position index.
    xsize : int
        Number of points to extract in the x direction.
    ysize : int
        Number of points to extract in the y direction.
    verbose : bool, optional
        If set to True, print progress (default is False).
        
    Returns:
    --------
    np.ndarray
        A 3D tensor of information extracted from the dataset.
    """

    bands, rows, cols = dataset.RasterCount, ysize, xsize
    output = np.empty((bands, rows, cols))
    
    for r_idx in range(bands):
        band = dataset.GetRasterBand(r_idx + 1)
        arr = band.ReadAsArray(x_off, y_off, cols, rows)
        output[r_idx, :, :] = arr
        if verbose:
            print(f"Processed band {r_idx + 1}/{bands}")
    
    return output

def open_datasets(file_paths: List[str]) -> Dict[str, gdal.Dataset]:
    """Open a list of files as GDAL datasets."""
    return {Path(fp).stem: gdal.Open(fp) for fp in file_paths}


def get_nps_(file_paths: List[str], verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Convert a list of GDAL dataset file paths to a dictionary of numpy arrays.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of file paths to GDAL datasets.
    verbose : bool, optional
        If set to True, print progress (default is False).
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with file names as keys and corresponding numpy arrays as values.
    """

    dsets = open_datasets(file_paths)
    
    nps = {
        fn: dataset_to_np(
            ds,
            x_off=0,
            y_off=0,
            xsize=ds.RasterXSize,
            ysize=ds.RasterYSize,
            verbose=verbose,
        )
        for fn, ds in tqdm(dsets.items())
    }
    
    return nps


def get_coords_res(dataset: gdal.Dataset) -> Dict[str, Union[float, int]]:
    """
    For given dataset, returns the position of the top-left corner and resolutions.

    Arguments:
    ----------
      dataset (osgeo.gdal.Dataset): gdal dataset

    Returns:
    -------
      dict: Contains coordinates of the top-left corner and
            resolutions along the x and y axes.
    """
    gt = dataset.GetGeoTransform()

    return {
        "x": gt[0],
        "y": gt[3],
        "x_res": gt[1],
        "y_res": gt[5]
    }


def extract_latitude_longitude(path_to_tifs: str, feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D arrays of longitude and latitude of given .tif data.
    Helps to build a mapping between raster spatial indices and coordinates on the earth.

    Arguments:
    ----------
      path_to_tifs (str): path to directory that contains terraclim dataset.
      feature_name (str): feature name of interest.

    Returns:
    -------
      tuple: Longitude and latitude 1D arrays.
    """
    file_paths = get_file_paths(path_to_tifs, [feature_name])
    dset_tmp = gdal.Open(file_paths[feature_name][0])
    coords_dict = get_coords_res(dset_tmp)

    Lat = coords_dict["y"] + np.arange(dset_tmp.RasterYSize) * coords_dict["y_res"]
    Lon = coords_dict["x"] + np.arange(dset_tmp.RasterXSize) * coords_dict["x_res"]

    return Lat, Lon


def get_unique_values(data: np.ndarray) -> List[Tuple[Union[int, float], int]]:
    """
    Returns a sorted list of unique values in the data and their respective counts.

    Arguments:
    ----------
      data (np.ndarray): Input data array.

    Returns:
    -------
      List[Tuple[Union[int, float], int]]: List of tuples where each tuple contains a unique value and its count.
    """
    
    unique, counts = np.unique(data, return_counts=True)
    values = [(u, c) for u, c in zip(unique, counts)]
    
    # Sort values based on counts
    values.sort(key=lambda tup: tup[1])

    return values


def get_class_distribution(data: Iterable[Union[int, float]]) -> None:
    """
    Displays the percentage distribution of each unique class in the data.

    Arguments:
    ----------
      data (Iterable[Union[int, float]]): Input data array or list.

    Returns:
    -------
      None
    """
    
    unique_values = get_unique_values(data)
    value_dict = dict(unique_values)
    total = sum(value_dict.values())
    
    for k, v in value_dict.items():
        print(f"Class {k}: {v / total * 100:.2f} %")


def get_target_data(path_to_raw_data: str, path_to_processed_data: str) -> None:
    """
    Processes target data from a raster file and saves it as a pickled dictionary.

    Arguments:
    ----------
    path_to_raw_data (str): Path to the raw raster data.
    path_to_processed_data (str): Path to save the processed data.

    Returns:
    -------
    None
    """
    
    with gdal.Open(path_to_raw_data) as dataset:
        # Save raster to np array
        np_data = dataset_to_np(
            dataset,
            x_off=0,
            y_off=0,
            xsize=dataset.RasterXSize,
            ysize=dataset.RasterYSize,
        )

        # Reshape data
        reshaped_data = np_data[0, :, :].reshape(-1)

        # Create and save the target dictionary
        target_dict = {"Target": reshaped_data}
        
        with open(path_to_processed_data, "wb") as f:
            pickle.dump(target_dict, f, protocol=4)


def get_features_data(path_to_raw_data: Dict[str, str], path_to_processed_data_folder: str) -> None:
    """
    Processes features data from a set of raster files and saves them as pickled dictionaries.

    Arguments:
    ----------
    path_to_raw_data (Dict[str, str]): Dictionary where key is the feature name and value is the path to its raw data.
    path_to_processed_data_folder (str): Directory path to save the processed data.

    Returns:
    -------
    None
    """
    
    monthly_keys = set([
        "fy", "pr", "pr_p95", "sfcWindmax", "snw", "t0ud", "tas", "tasmax", 
        "tasmin", "Tstep6", "tp", "t2m", "monT0ud", "monTstep6"
    ])
    
    morphological_keys = set(["morf_3", "morf_11", "morf_33", "morf_47"])
    
    other_keys = set(["DEM_1km", "12m_SPI"])

    for feature, path in path_to_raw_data.items():
        # Define file paths:
        file_paths = glob.glob(os.path.join(path, "*.tif"))

        # Create a dictionary to store feature data
        features_dict = dict()

        # Extract data from raster files
        data_dict = get_nps_(file_paths)

        # Process monthly data
        for key, data in data_dict.items():
            if key in monthly_keys:
                for month in range(12):
                    feature_key = f"{key}_M{month + 1}"
                    features_dict[feature_key] = data[month, :, :].reshape(-1)

            # Process morphological data
            elif key in morphological_keys:
                for i in range(1, 11):
                    feature_key = f"{key}_{i}"
                    features_dict[feature_key] = data[i - 1, :, :].reshape(-1)

            # Process other data (Relief and SPI)
            elif key in other_keys:
                features_dict[key] = data[0, :, :].reshape(-1)

        # Save the processed feature data
        output_path = os.path.join(path_to_processed_data_folder, f"features_{feature}.npy")
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f, protocol=4)


def check_dimensions(path_to_features, path_to_target)-> None:
    """
    Check if the first dimensions of the feature and target numpy files match.

    Arguments:
    - path_to_features (str): Path to the features numpy file.
    - path_to_target (str): Path to the target numpy file.

    Returns:
    None
    """
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
