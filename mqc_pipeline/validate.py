import numpy as np
import pandas as pd
from pathlib import Path

from .settings import ValidationError

CSV_COL_NAMES = ("smiles", "Smiles", "SMILES")

PathLike = str | Path


def is_csv_single_column(csv_path: PathLike, first_n_rows: int = 5) -> bool:
    """
    Checks if the CSV file has only one column and that column is named
    'smiles', 'Smiles', or 'SMILES'.

    :param csv_path: Path to the CSV file.
    :param first_n_rows: Number of rows to check for validation.
    """
    assert Path(
        csv_path).suffix == ".csv", "This function expects a .csv file path."
    try:
        # Read the header and the first few rows
        df = pd.read_csv(csv_path, nrows=first_n_rows)
        columns = df.columns.tolist()
        # Check if there's only one column and the column name is supported
        if len(columns) != 1 or columns[0] not in CSV_COL_NAMES:
            return False
        # Ensure all rows in the first column are non-empty
        return bool(df.iloc[:, 0].notna().all())
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def is_txt_single_column(txt_path: PathLike, first_n_rows: int = 5) -> bool:
    """
    Checks if the .txt file contains only a single column of non-empty strings.

    :param txt_path: Path to the .txt file.
    :param first_n_rows: Number of rows to check for validation.
    """
    assert Path(
        txt_path).suffix == ".txt", "This function expects a .txt file path."
    try:
        # Read only the first few rows
        data = np.loadtxt(txt_path,
                          dtype=str,
                          comments='#',
                          ndmin=1,
                          max_rows=first_n_rows)
        # Check if the data is a single column:
        # ndim == 1 for 1D array, 2D array with shape (n, 1)
        return data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def validate_input(input_file_or_dir: str):
    """
    Validate the input file or directory.
    """
    input_file_or_dir = Path(input_file_or_dir)
    # Validate single-file input (contains SMILES strings)
    if input_file_or_dir.is_file():
        if input_file_or_dir.suffix not in ['.txt', '.csv']:
            raise ValidationError("Input file must be a .txt or .csv file.")
        # Check if the file has a single column
        if input_file_or_dir.suffix == '.csv' and (
                not is_csv_single_column(input_file_or_dir)):
            raise ValidationError(
                "CSV file must contain a single column of smiles strings.")

        if input_file_or_dir.suffix == '.txt' and (
                not is_txt_single_column(input_file_or_dir)):
            raise ValidationError(
                "Text file must contain a single column of smiles strings.")
    # Validate directory input (XYZ files)
    elif input_file_or_dir.is_dir():
        # Check for the first xyz file only (stopping at first match)
        xyz_files = input_file_or_dir.glob("*.xyz")
        try:
            next(xyz_files)
        except StopIteration:
            raise ValidationError(
                "Directory must contain at least one .xyz file.")
    else:
        raise ValidationError(
            "Input must be a valid .txt, .csv file or directory containing xyz files."
        )

    return str(input_file_or_dir)
