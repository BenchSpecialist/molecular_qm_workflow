from pathlib import Path
import pytest

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from pyscf import M

from mqc_pipeline import smiles_to_3d_structures_by_rdkit, Structure
from mqc_pipeline.smiles_util import get_canonical_smiles
from mqc_pipeline.adaptors import get_adaptor


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


xyz_block = """9
ethanol
C       -0.7482   0.0156   0.0000
C        0.7482  -0.0156   0.0000
O        1.2848   1.2034   0.0000
H       -1.1187  -0.4862   0.8737
H       -1.1187  -0.4862  -0.8737
H       -1.1091   1.0241   0.0000
H        1.1187  -0.4862   0.8737
H        1.1187  -0.4862  -0.8737
H        1.8465   1.1830   0.7700
"""
canonical_smiles = "CCO"


def test_get_canonical_smiles_xyz_file(tmp_cwd):
    # Test with a txt file
    xyz_path = Path("test.xyz")
    xyz_path.write_text(xyz_block)
    assert get_canonical_smiles(str(xyz_path)) == canonical_smiles


def test_get_canonical_smiles_obj():
    # Test with RDKit Mol object
    rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
    assert get_canonical_smiles(rdkit_mol) == canonical_smiles

    # Test with a PySCF Mole object
    pyscf_mole = M(atom=xyz_block.lstrip('9\nethanol\n'),
                   basis='sto-3g',
                   unit='angstrom')
    assert get_canonical_smiles(pyscf_mole) == canonical_smiles

    # Test with a Structure object
    adaptor = get_adaptor(rdkit_mol)
    st = adaptor.to_structure()
    assert get_canonical_smiles(st) == canonical_smiles
