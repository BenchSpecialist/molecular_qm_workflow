from pathlib import Path

import pytest

from mqc_pipeline.validate import CSV_COL_NAMES, is_csv_single_column, \
    is_txt_single_column, validate_input, ValidationError


@pytest.mark.parametrize("col_name", CSV_COL_NAMES)
def test_is_csv_single_column(col_name, tmp_cwd):
    csv_path = "test.csv"
    csv_content = f"""{col_name}
    O=C1OCCO1
    CCOC(=O)OCC
    """
    Path(csv_path).write_text(csv_content)
    assert is_csv_single_column(csv_path, first_n_rows=2) is True

    # Test with a CSV file with multiple columns
    csv_content = f"""{col_name},other_col
    O=C1OCCO1,other_data
    """
    Path(csv_path).write_text(csv_content)
    assert is_csv_single_column(csv_path, first_n_rows=2) is False


def test_is_txt_single_column(tmp_cwd):
    txt_path = "test.txt"
    txt_content = """# Input data
    O=C1OCCO1
    CCOC(=O)OCC
    """
    Path(txt_path).write_text(txt_content)
    assert is_txt_single_column(txt_path, first_n_rows=2) is True
    # Test with a .txt file with multiple columns
    txt_content = """# Input data
    O=C1OCCO1 other_data
    CCOC(=O)OCC other_data
    """
    Path(txt_path).write_text(txt_content)
    assert is_txt_single_column(txt_path, first_n_rows=2) is False


def test_validate_input(tmp_cwd):
    inp_csv = "smiles.csv"
    with pytest.raises(ValidationError,
                       match="Input file or directory does not exist"):
        validate_input(inp_csv)

    # Create the input file and check if the input validation passes
    Path(inp_csv).write_text("smiles\nC\n")
    validate_input(inp_csv)

    inp_txt = "smiles.txt"
    Path(inp_txt).write_text("C")
    validate_input(inp_txt)

    # Test with invalid file format
    invalid_inp = "smiles.invalid"
    Path(invalid_inp).touch()
    with pytest.raises(ValidationError,
                       match="Input file must be a .txt or .csv file."):
        validate_input(invalid_inp)

    # Test with a directory
    xyz_dir = "input_xyz_dir"
    with pytest.raises(ValidationError,
                       match="Input file or directory does not exist"):
        validate_input(xyz_dir)

    Path(xyz_dir).mkdir(exist_ok=True)
    with pytest.raises(ValidationError,
                       match="Directory must contain at least one .xyz file"):
        validate_input(xyz_dir)

    # Create one .xyz file in the directory and check if the validation passes
    xyz_file = Path(xyz_dir) / "test.xyz"
    xyz_file.touch()
    validate_input(xyz_dir)
