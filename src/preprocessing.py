import os
import copy

import numpy as np
import rasterio
import rasterio.mask
import rasterio.plot
from shapely import geometry
from tqdm import tqdm
from typing import List, Dict, Any


class crop_features:
    """
    Represents the features required for cropping and reshaping.

    Attributes:
    -----------
    poly : shapely.geometry.Polygon
        Polygon representing the cropping area.
    transform : affine.Affine
        Transformation matrix for cropping and reshaping.
    height : int
        Height of the crop.
    width : int
        Width of the crop.
    """

    def __init__(self, left, top, right, bottom, width, height):
        p1 = geometry.Point(left, bottom)
        p2 = geometry.Point(left, top)
        p3 = geometry.Point(right, top)
        p4 = geometry.Point(right, bottom)

        pointList = [p1, p2, p3, p4]
        self.poly = geometry.Polygon([i for i in pointList])

        bbox_size = (height, width)
        bbox = [left, bottom, right, top]
        self.transform = rasterio.transform.from_bounds(  # type: ignore
            *bbox, width=bbox_size[1], height=bbox_size[0]
        )
        self.height = height
        self.width = width


def average_10years_climate(
    path_climate: str,
    path_new: str,
    years: List[int],
    months: List[str],
    folders: List[str],
    name_str: str,
) -> None:
    """
    Averages data for the same calendar month across specified years.
    Climate data is sourced from the required years.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.
    months: List[str]
        Month numbers.
    folders: List[str]
        Folders from which data needs to be retrieved.
    name_str: str
        Auxiliary string for file naming.

    Returns:
    --------
    None
        Writes averaged data to new .tiff files with 12 bands (1 band per each month).
    """
    for folder in tqdm(folders):
        path = os.path.join(path_climate, folder, "")
        filelist = os.listdir(path)

        # Random source file for meta data
        with rasterio.open(path + filelist[0]) as src:
            w = src.width
            h = src.height
            profile = src.profile
            profile.update({"count": 12, "dtype": "float32", "nodata": None})

        # Empty array to keep average  month values for 10 years
        array_avg = np.empty((h, w, 0))

        for month in months:
            # Empty array to keep 10 values (for each year)
            array = np.empty((h, w, 0))
            for year in years:
                try:
                    with rasterio.open(
                        path + folder + name_str + str(year) + "_" + month + ".tiff"
                    ) as src:
                        array = np.dstack((array, src.read(1))).astype("float32")
                except rasterio.errors.RasterioIOError:  # type: ignore
                    with rasterio.open(
                        path + folder + name_str + str(year) + "_" + month + ".tif"
                    ) as src:
                        array = np.dstack((array, src.read(1))).astype("float32")

            # Calculate average over all years
            array_avg = np.dstack((array_avg, np.mean(array, axis=2))).astype("float32")

        # Write the data to the output raster
        array_avg = array_avg.transpose(2, 0, 1)

        output_file = os.path.join(
            path_new, f"{folder}{name_str}{years[0]}_{years[-1]}.tif"
        )
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(array_avg)


def average_spi(
    path_climate: str,
    path_new: str,
    years: List[int],
    folders: List[str],
    name_str: str,
) -> None:
    """
    Averages SPI data across specified years.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.
    folders: List[str]
        Folders from which data needs to be retrieved.
    name_str: str
        Auxiliary string for file naming.

    Returns:
    --------
    None
        Writes averaged data to new .tiff files.
    """
    for folder in tqdm(folders):
        path = os.path.join(path_climate, folder, "")
        filelist = os.listdir(path)

        # Random source file for meta data
        with rasterio.open(os.path.join(path, filelist[0])) as src:
            w, h = src.width, src.height
            profile = src.profile
            profile.update(count=1)

        yearly_values = np.empty((h, w, 0))

        for year in years:
            file_name = os.path.join(path, f"{folder}{name_str}{year}_00.tiff")
            with rasterio.open(file_name) as src:
                yearly_values = np.dstack((yearly_values, src.read(1)))

        # Calculate average over all years
        avg_values = np.mean(yearly_values, axis=2).astype("float32")

        output_file = os.path.join(
            path_new, f"{folder}{name_str}{years[0]}_{years[-1]}.tif"
        )
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(avg_values, 1)


def process_past_climate(path_climate: str, path_new: str, years: List[int]) -> None:
    """
    Identifies relevant source folders and invokes the auxiliary function to average past climate data.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.

    Returns:
    --------
    None
        Invokes the `average_10years_climate` function that creates new .tiff files
            with 12 bands (1 band per month).
    """

    # Filter directories related to past climate data
    folders = os.listdir(path_climate)
    folders_past = [
        name for name in folders if "spi" not in name.lower() and "era" in name.lower()
    ]

    months = [f"{i:02d}" for i in range(1, 13)]

    average_10years_climate(path_climate, path_new, years, months, folders_past, "_")

    print("Past climate data is collected")


def process_future_climate(
    path_climate: str, path_new: str, years: List[int], ssps: List[str]
) -> None:
    """
    Identifies relevant source folders and invokes the auxiliary function to average future climate data.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.
    ssps: List[str]
        List of Shared Socioeconomic Pathways (SSPs) for the analysis.

    Returns:
    --------
    None
        Invokes the `average_10years_climate` function that creates new .tiff files
            with 12 bands (1 band per month).
    """

    # Filter directories related to future climate data
    folders = os.listdir(path_climate)
    folders_future = [
        name
        for name in folders
        if "spi" not in name.lower() and "era" not in name.lower()
    ]

    months = [f"{i:02d}" for i in range(1, 13)]

    for ssp in ssps:
        average_10years_climate(
            path_climate, path_new, years, months, folders_future, f"_{ssp}_"
        )

    print("Future climate data is collected")


def process_past_spi(path_climate: str, path_new: str, years: List[int]) -> None:
    """
    Identifies relevant source folders and invokes the auxiliary function to average past SPI data.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.

    Returns:
    --------
    None
        Invokes the `average_spi` function that creates new .tiff files 
            with 12 bands (1 band per month).
    """

    # Filter directories related to past SPI data
    folders = os.listdir(path_climate)
    folders_past_spi = [
        name for name in folders if "spi" in name.lower() and "era" in name.lower()
    ]

    average_spi(path_climate, path_new, years, folders_past_spi, "_")

    print("Past climate SPI data is collected")


def process_future_spi(
    path_climate: str, path_new: str, years: List[int], ssps: List[str]
) -> None:
    """
    Identifies relevant source folders and invokes the auxiliary function to average future SPI data.

    Parameters:
    -----------
    path_climate: str
        Path to the source folder.
    path_new: str
        Path to the destination folder.
    years: List[int]
        Years for which data is required.
    ssps: List[str]
        List of Shared Socioeconomic Pathways (SSPs) for the analysis.

    Returns:
    --------
    None
        Invokes the `average_spi` function that creates new .tiff files 
            with 12 bands (1 band per month).
    """

    # Filter directories related to future SPI data
    folders = os.listdir(path_climate)
    folders_future_spi = [
        name for name in folders if "spi" in name.lower() and "era" not in name.lower()
    ]

    for ssp in ssps:
        average_spi(path_climate, path_new, years, folders_future_spi, f"_{ssp}_")

    print("Future climate SPI data is collected")


def crop_tiff(path_src: str, path_dest: str, bound: Any) -> None:
    """
    Crops .tiff files based on a bounding polygon and writes new cropped files.

    Parameters:
    -----------
    path_src: str
        Path to the source directory containing the .tiff files to be cropped.
    path_dest: str
        Path to the destination directory where the cropped .tiff files will be saved.
    bound: Any
        An object with a `poly` attribute representing the bounding polygon for cropping.

    Returns:
    --------
    None
        Writes new cropped .tiff files to the destination directory.
    """
    files = os.listdir(path_src)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(path_dest, exist_ok=True)

    # Iterate over the files and apply the cropping
    for fname in tqdm(files):
        filepath = os.path.join(path_src, fname)

        with rasterio.open(filepath) as src:
            out_image, out_transform = rasterio.mask.mask(src, [bound.poly], crop=True)
            
            profile = src.profile
            profile.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            with rasterio.open(os.path.join(path_dest, fname), "w", **profile) as dst:
                dst.write(out_image)

    print(f"{len(files)} images cropped")


def reshape_tiff(path_src: str, path_dest: str, bound: Any) -> None:
    """
    Reshapes .tiff files based on given dimensions and transformation, then writes the new files.

    Parameters:
    -----------
    path_src: str
        Path to the source directory containing the .tiff files to be reshaped.
    path_dest: str
        Path to the destination directory where the reshaped .tiff files will be saved.
    bound: Any
        An object with attributes `height`, `width`, and `transform` that provide the new dimensions
        and transformation properties for the .tiff files.

    Returns:
    --------
    None
        Writes new reshaped .tiff files to the destination directory.
    """

    # List files in the source directory
    files = os.listdir(path_src)

    # Create the destination directory if it doesn't exist
    os.makedirs(path_dest, exist_ok=True)

    # Iterate over the files and apply the reshaping
    for fname in tqdm(files):
        filepath = os.path.join(path_src, fname)

        with rasterio.open(filepath) as src:
            profile = src.profile
            profile.update(
                {
                    "height": bound.height,
                    "width": bound.width,
                    "transform": bound.transform,
                }
            )

            with rasterio.open(os.path.join(path_dest, fname), "w", **profile) as dst:
                dst.write(src.read())

    print(f"{len(files)} images reshaped")


def rename_climate(path: str, ssps: List[str]) -> None:
    """
    Renames .tiff files based on specific conditions and organizes them into corresponding directories.

    Parameters:
    -----------
    path: str
        Path to the source directory containing the .tiff files to be renamed.
    ssps: List[str]
        List of Shared Socioeconomic Pathways (SSPs) to be used in the renaming and organization.

    Returns:
    --------
    None
        Renames and organizes the files in the given directory.
    """
    
    from_files = os.listdir(path)

    # Corresponding names
    to_ = [
        "fy",
        "tp",
        "pr_p95",
        "sfcWindmax",
        "snw",
        "12m_SPI",
        "monT0ud",
        "tasmax",
        "tasmin",
        "t2m",
        "monTstep6",
    ]
    from_ = [
        "fy",
        "tp",
        "pr_p95",
        "sfcWindmax",
        "snw",
        "spi",
        "T0",
        "tasmax",
        "tasmin",
        "t2m",
        "step",
    ]

    for file in from_files:
        if ("pr_" in file) and ("95" not in file):
            new_name = file.replace("pr", "tp")
            os.rename(path + file, path + new_name)
        elif ("tas" in file) and ("min" not in file) and ("max" not in file):
            new_name = file.replace("tas", "t2m")
            os.rename(path + file, path + new_name)

    for ssp in ssps:
        if ssp != "None":
            for folder in ["CMCC", "CNRM", "MRI"]:
                if not os.path.exists(path + "/" + ssp + "/" + folder):
                    os.makedirs(path + "/" + ssp + "/" + folder)

    from_files_ = os.listdir(path)
    if "None" not in ssps:
        [from_files_.remove(x) for x in ssps if x in from_files_]

    for ssp in ssps:
        for file in from_files_[:]:
            for i, item in enumerate(from_):
                if file.find(item) != -1:
                    index = i
            if "CMCC" in file and ssp in file:
                from_files_.remove(file)
                os.rename(path + file, path + ssp + "/CMCC/" + to_[index] + ".tif")  # type: ignore
            elif "CNRM" in file and ssp in file:
                from_files_.remove(file)
                os.rename(path + file, path + ssp + "/CNRM/" + to_[index] + ".tif")  # type: ignore
            elif "MRI" in file and ssp in file:
                from_files_.remove(file)
                os.rename(path + file, path + ssp + "/MRI/" + to_[index] + ".tif")  # type: ignore
            elif ssp not in file and all(
                substr not in file for substr in ["CMCC", "CNRM", "MRI"]
            ):
                os.rename(path + file, path + to_[index] + ".tif")

    print(f"Renaming and organization of files in {path} completed.")
