from pathlib import Path

import pytest
from mqc_pipeline.validate import CSV_COL_NAMES, is_csv_single_column, \
    is_txt_single_column


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
