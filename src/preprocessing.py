import os
import copy

import numpy as np
import rasterio
import rasterio.mask
import rasterio.plot
from shapely import geometry
from tqdm import tqdm


class crop_features:
    """Class holding features for cropping and reshape

    Returns:
    --------
        new object of the class
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


def average_10years_climate(path_climate, path_new, years, months, folders, name_str):
    """Averages data for the same calendar month along few years
    Climate data is taken from required 10 years

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years
        months: Callable[List[str]]
            Nubmers of months
        folders: Callable[List[str]]
            Folders to retrieve data from
        name_str: Callable[str]
            Auixilary string

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
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
        with rasterio.open(
            path_new
            + folder
            + name_str
            + str(years[0])
            + "_"
            + str(years[-1])
            + ".tif",
            "w",
            **profile,
        ) as dst:
            dst.write(array_avg)


def average_spi(path_climate, path_new, years, folders, name_str):
    """Averages SPI data along few years
    Data is taken from required 10 years

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years
        folders: Callable[List[str]]
            Folders to retrieve data from
        name_str: Callable[str]
            Auixilary string

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
    """
    for folder in tqdm(folders):
        path = os.path.join(path_climate, folder, "")
        filelist = os.listdir(path)

        # Random source file for meta data
        with rasterio.open(path + filelist[0]) as src:
            w = src.width
            h = src.height
            profile = src.profile
            profile.update(
                {
                    "count": 1,
                    # "dtype": "float64",
                    # "nodata": None
                }
            )

        # Empty array to keep average  month values for 10 years
        array_avg = np.empty((h, w, 0))

        for year in years:
            with rasterio.open(
                path + folder + name_str + str(year) + "_00.tiff"
            ) as src:
                array_avg = np.dstack((array_avg, src.read(1)))

        # Calculate average over all years
        array_avg = np.mean(array_avg, axis=2)

        # Write new tif
        with rasterio.open(path_new + folder + name_str + str(years[0]) + "_" + str(years[-1]) + ".tif", "w", **profile) as dst:  # type: ignore
            dst.write(array_avg.astype("float32"), 1)


def process_past_climate(path_climate, path_new, years):
    """Defines source folder and run auxilary funtion to average past data

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
    """
    # List all data folders and choose those related to past
    folders = os.listdir(path_climate)
    folders_past = [
        name
        for name in folders
        if (("spi" not in name) & (("ERA" in name) | ("Era" in name)))
    ]

    months = [f"{i:02d}" for i in range(1, 13)]
    average_10years_climate(path_climate, path_new, years, months, folders_past, "_")
    print("Past climate data is collected")


def process_future_climate(path_climate, path_new, years, ssps):
    """Defines source folder and run auxilary funtion to average future data

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
    """
    # List all data folders and choose those related to future
    folders = os.listdir(path_climate)
    folders_future = [
        name
        for name in folders
        if (("spi" not in name) & ("ERA" not in name) & ("Era" not in name))
    ]

    months = [f"{i:02d}" for i in range(1, 13)]
    for ssp in ssps:
        average_10years_climate(
            path_climate, path_new, years, months, folders_future, "_" + ssp + "_"
        )
    print("Future climate data is collected")


def process_past_spi(path_climate, path_new, years):
    """Defines source folder and run auxilary funtion to average past SPI data

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
    """
    # List all data folders and choose those related to past
    folders = os.listdir(path_climate)
    folders_past_spi = [
        name
        for name in folders
        if (("spi" in name) & (("ERA" in name) | ("Era" in name)))
    ]
    average_spi(path_climate, path_new, years, folders_past_spi, "_")
    print("Past climate SPI data is collected")


def process_future_spi(path_climate, path_new, years, ssps):
    """Defines source folder and run auxilary funtion to average future SPI data

    Parameters:
    --------
        path_climate: Callable[str]
            Source folder
        path_new: Callable[Dict]
            Destination folder
        years: Callable[List[int]]
            Required years

    Returns:
    --------
        new .tiff files with 12 bands (1 band per each month)
    """
    # List all data folders and choose those related to future
    folders = os.listdir(path_climate)
    folders_future_spi = [
        name
        for name in folders
        if (("spi" in name) & ("ERA" not in name) & ("Era" not in name))
    ]
    for ssp in ssps:
        average_spi(path_climate, path_new, years, folders_future_spi, "_" + ssp + "_")

    print("Future climate SPI data is collected")


def crop_tiff(path_src, path_dest, bound):
    """Crops .tiff file and writes new one

    Parameters:
    --------
        path_src: Callable[str]
            Source folder
        path_dest: Callable[Dict]
            Destination folder
        bound: Callable[Object]
            Transformation properties

    Returns:
    --------
        new cropped .tiff files
    """
    files = os.listdir(path_src)
    files = [name for name in files if not ("filter" in name)]
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    # Apply those parameters for transformation
    for fname in tqdm(files):
        filepath = path_src + fname

        with rasterio.open(filepath) as src:
            out_image, out_transform = rasterio.mask.mask(src, [bound.poly], crop=True)
            # Create a new cropped raster to write to
            profile = src.profile
            profile.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            with rasterio.open(path_dest + fname, "w", **profile) as dst:
                # Read the data and write it to the output raster
                dst.write(out_image)
    print(len(files), "images croped")


def reshape_tiff(path_src, path_dest, bound):
    """Reshapes .tiff file and writes new one

    Parameters:
    --------
        path_src: Callable[str]
            Source folder
        path_dest: Callable[Dict]
            Destination folder
        bound: Callable[Object]
            Transformation properties

    Returns:
    --------
        new cropped .tiff files
    """
    files = os.listdir(path_src)

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    # Apply those parameters for transformation
    for fname in tqdm(files):
        filepath = path_src + fname

        with rasterio.open(filepath) as src:
            # Create a new cropped raster to write to
            profile = src.profile
            profile.update(
                {
                    "height": bound.height,
                    "width": bound.width,
                    "transform": bound.transform,
                }
            )

            with rasterio.open(path_dest + fname, "w", **profile) as dst:
                # Read the data and write it to the output raster
                dst.write(src.read())
    print(len(files), "images reshaped")


def rename_climate(path, ssps):
    """Renames .tiff file and moves it

    Parameters:
    --------
        path: Callable[str]
            Source folder

    Returns:
    --------
        Same files being renamed
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
                os.rename(path + file, path + to_[index] + ".tif")  # type: ignore
