import pytest

from mqc_pipeline import smiles_to_3d_structures_by_rdkit


def test_smiles_to_3d_structures_by_rdkit():
    # Test with a valid SMILES string
    smiles = "C"
    st = smiles_to_3d_structures_by_rdkit(smiles)
    assert st.smiles == smiles
    assert st.elements == ['C', 'H', 'H', 'H', 'H']
    assert st.xyz.shape == (5, 3)

    # Raise error when given an invalid SMILES string
    invalid_smiles = "INVALID_SMILES"
    with pytest.raises(ValueError, match='Invalid SMILES string'):
        smiles_to_3d_structures_by_rdkit(invalid_smiles)
