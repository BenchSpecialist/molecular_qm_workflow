from rdkit import Chem
from pyscf import M

from mqc_pipeline import Structure
from mqc_pipeline.adaptors import get_adaptor
from mqc_pipeline.common import get_unpaired_electrons


def test_StructureAdaptor():
    elements = ['O', 'H']
    xyz = [[0, 0, 0], [0.97, 0, 0]]
    charge = -1
    multiplicity = 1
    st = Structure(elements=elements,
                   xyz=xyz,
                   charge=charge,
                   multiplicity=multiplicity)
    st_adaptor = get_adaptor(st)

    # Test conversion to RDKit Mol object
    rdkit_mol = st_adaptor.to_rdkit_mol()
    assert isinstance(rdkit_mol, Chem.Mol)
    # Test that H atoms are not removed by default
    assert rdkit_mol.GetNumAtoms() == len(elements)
    # Test that bonding information is generated
    assert rdkit_mol.GetNumBonds() == 1
    assert get_unpaired_electrons(rdkit_mol) == multiplicity - 1

    pyscf_mole = st_adaptor.to_pyscf_mole(basis="sto-3g")
    assert pyscf_mole.charge == charge
    assert pyscf_mole.spin == multiplicity - 1
    assert pyscf_mole.basis == "sto-3g"


def test_RDKitMolAdaptor():
    rdkit_mol = Chem.MolFromSmiles('[OH-]')
    Chem.AllChem.EmbedMolecule(rdkit_mol)
    adaptor = get_adaptor(rdkit_mol)

    st = adaptor.to_structure()
    assert st.xyz.shape == (2, 3)
    assert st.elements == ["O", "H"]
    assert st.charge == -1
    assert st.multiplicity == 1

    pyscf_mole = adaptor.to_pyscf_mole(basis="sto-3g")
    assert pyscf_mole.charge == -1
    assert pyscf_mole.multiplicity == 1
    assert pyscf_mole.basis == "sto-3g"

    # Test the idempotency of the conversion
    rdkit_mol = adaptor.to_rdkit_mol(remove_hydrogens=True)
    assert rdkit_mol.GetNumAtoms() == 1
    rdkit_mol = adaptor.to_rdkit_mol(remove_hydrogens=False)
    assert rdkit_mol.GetNumAtoms() == 2


def test_PySCFMoleAdaptor():
    OH_anion = """
O     0.000000    0.000000    0.000000
H     0.970000    0.000000    0.000000
"""
    pyscf_mole = M(atom=OH_anion,
                   basis='6311g*',
                   unit='angstrom',
                   charge=-1,
                   spin=0)
    adaptor = get_adaptor(pyscf_mole)
    st = adaptor.to_structure()
    assert st.multiplicity == 1

    rdkit_mol = adaptor.to_rdkit_mol()
    # Bonding information is resolved
    assert rdkit_mol.GetNumBonds() == 1
