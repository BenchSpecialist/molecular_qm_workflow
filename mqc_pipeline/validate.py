import polars
from pathlib import Path
from rdkit import Chem

from .settings import ValidationError

SMILES_COL_NAMES = ("smiles", "Smiles", "SMILES", "smile", "SMILE",
                    "canonical_smile", "smi")

PathLike = str | Path


def validate_csv(csv_path: PathLike, first_n_rows: int = 5) -> bool:
    """
    Validate if the CSV file contains a SMILES string column with an acceptable
    column names.

    :param csv_path: Path to the CSV file.
    :param first_n_rows: Number of rows to check for validation.
    """
    assert Path(
        csv_path).suffix == ".csv", "This function expects a .csv file path."
    try:
        # Read the header and the first few rows
        df = polars.read_csv(csv_path, n_rows=first_n_rows, comment_prefix='#')
        # Check if the supported SMILES column names are present
        if not (smiles_col_name := next(
            (col for col in SMILES_COL_NAMES if col in df.columns), None)):
            raise ValidationError(
                f"No SMILES column found in CSV file. Acceptable column names: {', '.join(SMILES_COL_NAMES)}."
            )

        df = df.filter((polars.col(smiles_col_name).is_not_null())
                       & (polars.col(smiles_col_name) != ''))
        for smiles in df[smiles_col_name].to_list():
            if Chem.MolFromSmiles(smiles) is None:
                raise ValidationError(
                    f"Invalid SMILES string found in column '{smiles_col_name}': {smiles}"
                )

        # Check if all rows in the first column are non-empty strings
        # col_data = df.select(df.columns[0]).to_series()
        # return col_data.is_not_null().all() and (col_data.str.len_chars()
        #                                          > 0).all()
    except Exception as e:
        raise ValidationError(f"Error reading CSV: {e}")


def is_txt_single_column(txt_path: PathLike, first_n_rows: int = 5) -> bool:
    """
    Checks if the .txt file contains only a single column of non-empty strings.

    :param txt_path: Path to the .txt file.
    :param first_n_rows: Number of rows to check for validation.
    """
    txt_path = Path(txt_path)
    assert txt_path.suffix == ".txt", "This function expects a .txt file path."
    for line in txt_path.read_text().splitlines()[:first_n_rows]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # Skip empty lines and comments

        tokens = line.split()
        if len(tokens) != 1:
            return False  # More than one column

        smiles = tokens[0]
        if Chem.MolFromSmiles(smiles) is None:
            return False  # Invalid SMILES

    return True


def validate_input(input_file_or_dir: PathLike) -> str:
    """
    Validate the input file or directory.
    """
    input_file_or_dir = Path(input_file_or_dir)
    if not Path(input_file_or_dir).exists():
        raise ValidationError(
            f"Input file or directory does not exist: {input_file_or_dir}")

    # Validate single-file input (contains SMILES strings)
    if input_file_or_dir.is_file():
        if input_file_or_dir.suffix not in ['.txt', '.csv']:
            raise ValidationError("Input file must be a .txt or .csv file.")
        # Check if the file has a single column
        if input_file_or_dir.suffix == '.csv':
            validate_csv(input_file_or_dir)

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
