import numpy as np
from rdkit import Chem

from mqc_pipeline import Structure
from mqc_pipeline.common import get_unpaired_electrons_from_rdkit_mol


def test_Structure():
    # Test initialization with minimal parameters
    elements = ['O', 'H', 'H']
    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    st = Structure(elements=elements, xyz=xyz)
    assert st.elements == elements
    assert np.allclose(st.xyz, xyz)

    # Test that unique_id is generated and check its length
    assert st.unique_id is not None
    assert len(st.unique_id) == 11
    assert st.property == {}
    assert st.metadata == {}

    # Test the structural equality
    assert st == Structure(elements=elements, xyz=xyz)


def test_to_rdkit_mol():
    # Test conversion to RDKit Mol object
    elements = ['O', 'H']
    xyz = [[0, 0, 0], [0.97, 0, 0]]
    charge = -1
    multiplicity = 1
    st = Structure(elements=elements,
                   xyz=xyz,
                   charge=charge,
                   multiplicity=multiplicity)
    mol = st.to_rdkit_mol()
    assert isinstance(mol, Chem.Mol)
    # Test that H atoms are not removed by default
    assert mol.GetNumAtoms() == len(elements)
    # Test that bonding information is generated
    assert mol.GetNumBonds() == 1
    assert get_unpaired_electrons_from_rdkit_mol(mol) == multiplicity - 1
