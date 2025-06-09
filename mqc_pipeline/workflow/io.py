import polars
from pathlib import Path
from typing import Generator

from ..common import Structure
from ..structure_io import read_xyz
from ..util import get_default_logger

logger = get_default_logger()

CSV_COL_NAMES = ("smiles", "Smiles", "SMILES")


def read_smiles(input_file: str) -> list[str]:
    """
    Read SMILES strings from a single-column txt or csv file.
    """
    input_file = Path(input_file)
    if input_file.suffix == '.csv':
        df = polars.read_csv(input_file)

        if not any(col in df.columns for col in CSV_COL_NAMES):
            raise ValueError(
                "CSV file must contain a column named 'smiles', 'Smiles', or 'SMILES'."
            )

        for smiles_col_name in CSV_COL_NAMES:
            if smiles_col_name in df.columns:
                smiles_list = df[smiles_col_name].to_list()

    elif input_file.suffix == '.txt':
        smiles_list = [
            line.strip() for line in input_file.read_text().splitlines()
            if line.strip() and not line.startswith('#')
        ]
    else:
        raise ValueError(
            "Input file must be a .txt or .csv file containing a single column of smiles strings."
        )
    return smiles_list


def read_xyz_dir(input_dir: str) -> Generator[Structure, None, None]:
    """
    Read XYZ files from a directory, yield Structure objects one at a time.

    Usage:
    >>> for st in read_xyz_dir(input_dir):
    >>>     process_one_molecule(st)
    >>> full_sts_list = list(read_xyz_dir(input_dir))
    """
    for file in Path(input_dir).glob('*.xyz'):
        try:
            yield read_xyz(file)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
