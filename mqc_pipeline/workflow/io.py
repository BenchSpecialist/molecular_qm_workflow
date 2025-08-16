import polars
from pathlib import Path
from typing import Generator

from ..common import Structure
from ..structure_io import read_xyz
from ..util import get_default_logger
from ..validate import SMILES_COL_NAMES

logger = get_default_logger()


def read_smiles(input_file: str) -> list[str]:
    """
    Read SMILES strings from a single-column txt or csv file.

    :param input_file: Path to the input file (CSV or TXT).
    :return: List of SMILES strings.
    """
    input_file = Path(input_file)
    if input_file.suffix in ['.csv', '.parquet', '.parq']:
        # Get DataFrame from CSV or Parquet file
        if input_file.suffix == '.csv':
            df = polars.read_csv(input_file, comment_prefix='#')
        else:
            import pyarrow.parquet as pq
            table = pq.read_table(input_file)
            df = polars.from_arrow(table)
        # Check for SMILES column
        if not (smiles_col_name := next(
            (col for col in SMILES_COL_NAMES if col in df.columns), None)):
            raise ValueError(
                f"No SMILES column found in '{input_file}'. Allowed colunm names: {', '.join(SMILES_COL_NAMES)}."
            )
        # filter out None, empty strings.
        df = df.filter((polars.col(smiles_col_name).is_not_null())
                       & (polars.col(smiles_col_name) != ''))
        logger.info(
            f"Reading {df.height} SMILES strings from column '{smiles_col_name}'."
        )
        return df[smiles_col_name].to_list()
    elif input_file.suffix == '.txt':
        smiles_list = [
            line.strip() for line in input_file.read_text().splitlines()
            if line.strip() and not line.startswith('#')
        ]
        logger.info(
            f"Reading {len(smiles_list)} SMILES strings from {input_file}.")
        return smiles_list
    elif input_file.suffix == '.pkl':
        import pickle
        with open(input_file, 'rb') as f:
            smiles_list = pickle.load(f)
            logger.info(
                f"Reading {len(smiles_list)} SMILES strings from {input_file}."
            )
        if not isinstance(smiles_list, list) or not all(
                isinstance(smiles, str) for smiles in smiles_list):
            raise ValueError(
                "Pickle file must contain a list of SMILES strings.")
        return smiles_list
    else:
        raise ValueError(
            f"Unsupported file format: {input_file.suffix}. Only .csv, .txt, .pkl, and .parquet files are supported."
        )


def read_xyz_dir(input_dir: str) -> Generator[Structure, None, None]:
    """
    Read XYZ files from a directory, yield Structure objects one at a time.

    Usage:
    >>> for st in read_xyz_dir(input_dir):
    >>>     process_one_molecule(st)
    >>> full_sts_list = list(read_xyz_dir(input_dir))
    """
    for xyz_file in Path(input_dir).glob('*.xyz'):
        try:
            st = read_xyz(xyz_file, parse_comment=True)
            st.unique_id = xyz_file.stem  # Use the file name as the unique ID
            yield st
        except Exception as e:
            logger.error(f"Error parsing {xyz_file}: {e}")


def write_xyz_dir_from_csv(csv_path: str | Path,
                           output_dir: str | Path,
                           extended_xyz: bool = True) -> int:
    """
    Write XYZ files from a CSV file containing atom-level data.

    The CSV file must contain the following columns:
    - unique_id: Unique identifier for the molecule
    - element: Chemical element symbol (e.g., 'C', 'O', 'H')
    - x, y, z: Coordinates of the atom in Angstroms
    - smiles: Optional column containing the SMILES string of the molecule
    - Additional columns: Optional, will be included in the extended XYZ format
    :param csv_path: Path to the input CSV file.
    :param output_dir: Directory to save the output XYZ files.
    :param extended_xyz: If True, include additional columns in the XYZ file.

    :return: Number of XYZ files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = polars.read_csv(csv_path)

    default_cols = ['unique_id', 'element', 'x', 'y', 'z']
    if not all(col in df.columns for col in default_cols):
        raise ValueError(
            f"CSV file must contain the following columns: {', '.join(default_cols)}"
        )

    if extended_xyz:
        additional_cols = set(df.columns) - set(default_cols) - {'smiles'}
        additional_cols = sorted(list(additional_cols))

    # Group by unique ID
    unique_mol_ids = df["unique_id"].unique().to_list()
    mol_data_dict = {
        unique_id: df.filter(polars.col("unique_id") == unique_id)
        for unique_id in unique_mol_ids
    }
    num_xyz = 0
    for unique_id, mol_df in mol_data_dict.items():
        xyz_path = output_dir / f"{unique_id}.xyz"

        comment = f'{mol_df["smiles"].to_list()[0]}'
        if extended_xyz:
            comment += f' unique_id:{unique_id}'
            comment += f' AddtionalColumns:{",".join(additional_cols)}'

        lines = [f"{mol_df.shape[0]}", f"{comment}"]

        # Write atom coordinates and forces
        for row in mol_df.rows(named=True):
            element = row["element"]
            x, y, z = row["x"], row["y"], row["z"]
            line = f"{element:<2} {x:>15.6f} {y:>15.6f} {z:>15.6f}"

            if extended_xyz:
                for col in additional_cols:
                    line += f" {row[col]:>15.6f}"
            lines.append(line)

        xyz_path.write_text("\n".join(lines))
        num_xyz += 1

    return num_xyz


def get_sts_from_file(csv_or_parq: str, start_idx: int,
                      batch_size: int) -> list[Structure]:
    """
    Get a list of Structure objects from a CSV or Parquet file, with the specified
    range of rows.
    The input file must contain a column named 'xyz_block' with XYZ block strings,
    and optionally a column named 'smiles' with SMILES strings.

    :param csv_or_parq: Path to the input CSV or Parquet file.
    :param start_idx: Starting index of the rows to read.
    :param batch_size: Number of rows to read.

    :return: List of Structure objects.
    """
    XYZ_COL_NAME = "xyz_block"
    SMILES_COL_NAME = "smiles"

    infile = Path(csv_or_parq)
    if infile.suffix not in ['.csv', '.parquet', '.parq']:
        raise SystemExit(
            f"Input file must be a CSV or Parquet file, got: {infile}")

    df = polars.read_csv(
        infile) if infile.suffix == '.csv' else polars.read_parquet(infile)
    if XYZ_COL_NAME not in df.columns:
        raise SystemExit(f"Input file must contain '{XYZ_COL_NAME}' column.")

    # Read the specified range of rows
    df = df.slice(offset=start_idx, length=batch_size)
    sts = []
    for row in df.rows(named=True):
        st = Structure.from_xyz_block(row[XYZ_COL_NAME])
        st.smiles = row.get(SMILES_COL_NAME, None)
        sts.append(st)

    return sts
