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


def test_StructureAdaptor_to_rdkit_mol():
    """
    Covers an edge case "FS(F)(F)F" found on zinc20 dataset where using `Chem.MolFromXYZBlock`
    and `rdDetermineBonds.DetermineBonds` to get RDKit mol from XYZ gets an unexpected
    +2 overall molecular charge.
    """
    st = Structure(elements=['F', 'S', 'F', 'F', 'F'],
                   xyz=[[-1.28667578, 0.93860566, 0.66395486],
                        [0.16922749, 0.41059547, -0.02343354],
                        [1.54222291, -0.31828686, -0.69721161],
                        [-0.70176169, -0.40361351, -1.0819961],
                        [0.27823344, -0.63883451, 1.17232951]],
                   atomic_numbers=[9, 16, 9, 9, 9],
                   smiles='FS(F)(F)F',
                   unique_id='123',
                   charge=0,
                   multiplicity=1)
    st_adaptor = get_adaptor(st)
    rdmol = st_adaptor.to_rdkit_mol(remove_hydrogens=False)
    assert Chem.GetFormalCharge(rdmol) == 0

    Structure(elements=['C', 'S', 'O', 'F', 'H', 'H', 'H'],
              xyz=[[-0.80968252, -0.0393764, -0.02336478],
                   [0.92037434, 0.02538686, 0.47803707],
                   [1.66112181, -1.19306369, -0.03603409],
                   [1.68526824, 1.41279076, -0.09680693],
                   [-1.00166893, -0.97776898, -0.56486394],
                   [-1.03001311, 0.84409339, -0.65651269],
                   [-1.42539983, -0.07206194, 0.89954536]],
              smiles='C[S@@](=O)F',
              unique_id='456',
              charge=0,
              multiplicity=1)
    st_adaptor = get_adaptor(st)
    rdmol = st_adaptor.to_rdkit_mol(remove_hydrogens=False)


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
