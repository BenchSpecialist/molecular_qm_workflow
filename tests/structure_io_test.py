import pytest
import numpy as np
from mqc_pipeline import structure_io


@pytest.mark.parametrize("file_format", ["json", "pickle", "hdf5", "xyz"])
def test_single_structure(file_format, methane_st, tmp_cwd):
    file_path = f"one_st.{file_format}"

    # Test file writing can be executed without error
    structure_io.write(methane_st, file_path, format=file_format)

    # Read the structure back from the file
    structure_loaded = structure_io.read(file_path, format=file_format)

    # Check that the original and loaded structure are equal
    if file_format == "xyz":
        # xyz format loses some information in the conversion, so we only check
        # relevant attributes
        assert structure_loaded.elements == methane_st.elements
        assert structure_loaded.unique_id == methane_st.unique_id
        assert np.allclose(structure_loaded.xyz, methane_st.xyz)
    else:
        assert methane_st == structure_loaded


@pytest.mark.parametrize("file_format", ["json", "pickle", "hdf5"])
def test_multiple_structures(file_format, methane_st, n2_st, tmp_cwd):
    file_path = f"multi_sts.{file_format}"

    test_sts = [methane_st, n2_st]

    # Test writing multiple structures finishes without error
    structure_io.write(test_sts, file_path, format=file_format)

    # Read multiple structures from the file
    sts_loaded = structure_io.read(file_path, format=file_format)

    # Check that the original and loaded structures are equal
    assert test_sts == sts_loaded
