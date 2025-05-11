import pytest
import numpy as np
from mqc_pipeline import structure_io, Structure
from mqc_pipeline.property import DFT_ENERGY_KEY


def test_xyz_io(tmp_cwd):
    st = Structure(elements=['B', 'O', 'O', 'B', 'O'],
                   xyz=[[1.247458e+00, -4.614600e-02, -4.113860e-01],
                        [2.393653e+00, 3.140100e-01, -3.192300e-01],
                        [2.048000e-03, -4.888790e-01, -5.246490e-01],
                        [-1.248708e+00, -6.005700e-02, -4.180550e-01],
                        [-2.399377e+00, 2.872400e-01, -3.322480e-01]],
                   atomic_numbers=None,
                   smiles='B(=O)OB=O',
                   unique_id='54355245865',
                   charge=0,
                   multiplicity=1,
                   property={'energy_hartree': -275.62039364954484})

    xyz_file = "test.xyz"
    # Write the structure to an XYZ file
    structure_io.write_xyz(st, xyz_file)

    # Read the structure back from the XYZ file
    st_loaded = structure_io.read_xyz(xyz_file, parse_comment=True)

    # xyz format loses some information in the conversion, so we only check
    # relevant attributes
    assert np.allclose(st_loaded.xyz, st.xyz)
    if energy := st_loaded.property.get(DFT_ENERGY_KEY, None):
        assert np.isclose(energy, st.property[DFT_ENERGY_KEY])

    for attr in ("elements", "unique_id", "smiles", "charge", "multiplicity"):
        if loaded_val := getattr(st_loaded, attr):
            assert loaded_val == getattr(
                st, attr), f"Attribute {attr} does not match"


@pytest.mark.parametrize("file_format", ["json", "pickle"])
def test_single_structure_io(file_format, methane_st, tmp_cwd):
    file_path = f"one_st.{file_format}"

    # Test file writing can be executed without error
    structure_io.write(methane_st, file_path, format=file_format)

    # Read one structure back from the file
    st_loaded = structure_io.read(file_path, format=file_format)

    # Check that the original and loaded structure are equal
    assert methane_st == st_loaded


@pytest.mark.parametrize("file_format", ["json", "pickle"])
def test_multiple_structures_io(file_format, methane_st, n2_st, tmp_cwd):
    file_path = f"multi_sts.{file_format}"

    sts_list = [methane_st, n2_st]

    # Test writing multiple structures finishes without error
    structure_io.write(sts_list, file_path, format=file_format)

    # Read multiple structures from the file
    loaded_sts_list = structure_io.read(file_path, format=file_format)

    # Test that the original and loaded structures are equal
    assert sts_list == loaded_sts_list
