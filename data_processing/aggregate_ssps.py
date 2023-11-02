import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List

# Constants
DATA_FOLDER_NAME = os.path.join("data", "npys_data")
FEATURES_FOLDER_NAME = "2040_2050"
NPY_FILE_EXTENSION = "*.npy"

def group_files_by_ssp(path: Path) -> Dict[str, List[Path]]:
    """
    Groups .npy files in the given directory by their ssp values.

    :param path: Path of the directory containing .npy files.
    :return: Dictionary grouping file paths by ssp values.
    """
    ssp_groups = {}
    for file_path in path.glob(NPY_FILE_EXTENSION):
        ssp_value = file_path.name.split('features_')[1][:6]
        ssp_groups.setdefault(ssp_value, []).append(file_path)
    return ssp_groups

def average_data_for_ssp(ssp_files: List[Path], output_path: Path):
    """
    Loads data from files, averages it, and saves it as a new file.

    :param ssp_files: List of file paths to load data from.
    :param output_path: Path to save the averaged data file.
    """
    loaded_data = [np.load(file, allow_pickle=True) for file in ssp_files]

    if not all(isinstance(data, dict) for data in loaded_data):
        print(f"Files in {output_path.stem} group have non-dict values!")
        return

    keys = loaded_data[0].keys()
    if not all(data.keys() == keys for data in loaded_data):
        raise ValueError(f"Dictionaries in {output_path.stem} don't have matching keys!")

    averaged_data = {key: np.mean([data[key] for data in loaded_data], axis=0) for key in keys}
    
    with open(output_path, 'wb') as f:
        pickle.dump(averaged_data, f)
    print(f"Saved averaged data for {output_path.stem} to {output_path}")

def main():
    base_path = Path("..", DATA_FOLDER_NAME, FEATURES_FOLDER_NAME)
    ssp_groups = group_files_by_ssp(base_path)

    for ssp_value, ssp_files in ssp_groups.items():
        output_file = base_path / f"features_{ssp_value}_AVG.npy"
        average_data_for_ssp(ssp_files, output_file)

    print("Processing complete.")

if __name__ == "__main__":
    main()