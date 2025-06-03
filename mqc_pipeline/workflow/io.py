import polars
from pathlib import Path
from typing import Generator

from ..common import Structure
from ..structure_io import read_xyz
from ..util import get_default_logger

logger = get_default_logger()


def read_smiles(input_file: str) -> list[str]:
    """
    Read SMILES strings from a single-column txt or csv file.
    """
    input_file = Path(input_file)
    if input_file.suffix == '.csv':
        df = polars.read_csv(input_file)
        if 'smiles' not in df.columns:
            raise ValueError("CSV file must contain a 'smiles' column.")
        smiles_list = df['smiles'].to_list()
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
