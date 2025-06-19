from pathlib import Path
import pytest

from rdkit import Chem
from pyscf import M

from mqc_pipeline.smiles_util import smiles_to_structure_rdk, \
    get_canonical_smiles_rdk, get_canonical_smiles_ob, smiles_has_broken_bonds,\
    smiles_to_structure_pybel, generate_optimized_rdk_confs
from mqc_pipeline.adaptors import get_adaptor
from mqc_pipeline.test_util import requires_openbabel


def test_smiles_to_structure_rdk():
    # Test with a valid SMILES string
    smiles = "C"
    st = smiles_to_structure_rdk(smiles)
    assert st.smiles == smiles
    assert st.elements == ['C', 'H', 'H', 'H', 'H']
    assert st.xyz.shape == (5, 3)

    # Raise error when given an invalid SMILES string
    invalid_smiles = "INVALID_SMILES"
    with pytest.raises(ValueError, match='Invalid SMILES string'):
        smiles_to_structure_rdk(invalid_smiles)


def test_get_canonical_smiles_rdk(tmp_cwd):
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

    # Test with xyz file
    xyz_path = Path("test.xyz")
    xyz_path.write_text(xyz_block)
    assert get_canonical_smiles_rdk(str(xyz_path)) == canonical_smiles

    # Test with RDKit Mol object
    rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
    assert get_canonical_smiles_rdk(rdkit_mol) == canonical_smiles

    # Test with a PySCF Mole object
    pyscf_mole = M(atom=xyz_block.lstrip('9\nethanol\n'),
                   basis='sto-3g',
                   unit='angstrom')
    assert get_canonical_smiles_rdk(pyscf_mole) == canonical_smiles

    # Test with a Structure object
    adaptor = get_adaptor(rdkit_mol)
    st = adaptor.to_structure()
    assert get_canonical_smiles_rdk(st) == canonical_smiles


@pytest.mark.parametrize("input_smiles, has_broken_bonds", [
    ("CCCOC[C]F.FF", True),
    ("FC1COC(=O)O1", False),
    ("[CH2]OC(=O)O[CH]F", False),
    ("CCCOCC(F)(F)F", False),
])
def test_smiles_has_broken_bonds(input_smiles, has_broken_bonds):
    """
    Test the smiles_has_broken_bonds function with various SMILES strings.
    """
    assert smiles_has_broken_bonds(input_smiles) == has_broken_bonds


xyz1 = """10
4-fluoro-1,3-dioxolan-2-one, O1C(COC1=O)F, Energy: 34.2511096
O         -4.39245        0.67279       -0.03103
C         -3.98744       -0.64072       -0.03089
C         -2.44636       -0.57557       -0.00967
H         -4.35747       -1.15126       -0.94585
F         -4.47074       -1.27660        1.09918
O         -2.15598        0.76806        0.00173
H         -2.01496       -1.04166       -0.92151
H         -2.02943       -1.05635        0.90138
C         -3.30535        1.44926       -0.01134
O         -3.35738        2.66794       -0.00487
"""
can_smiles1 = "FC1COC(=O)O1"

xyz2 = """10
fluoromethyl methyl carbonate, O([CH]F)C(=O)O[CH2], Energy: 13.8613863
O         -4.49381        0.70538       -0.06415
C         -4.74670       -0.64889       -0.07273
C         -1.72854       -0.53571       -0.01002
H         -4.27983       -1.15892       -0.93798
F         -4.42286       -1.22738        1.14005
O         -2.07144        0.79695        0.01216
H         -2.05504       -1.02788       -0.94763
H         -2.05954       -1.05727        0.90900
C         -3.30657        1.37554       -0.00370
O         -3.35323        2.59408        0.02215
"""
can_smiles2 = "[CH2]OC(=O)O[CH]F"


@requires_openbabel()
@pytest.mark.parametrize("xyz_block, expected_smiles", [(xyz1, can_smiles1),
                                                        (xyz2, can_smiles2)],
                         ids=['xyz1', 'xyz2'])
def test_get_canonical_smiles_ob(xyz_block, expected_smiles, tmp_cwd):
    """
    Test the get_canonical_smiles_ob function with various XYZ blocks.
    """
    # Test with the XYZ block directly
    assert get_canonical_smiles_ob(xyz_block) == expected_smiles

    # Test with the XYZ file
    xyz_file = "test.xyz"
    Path(xyz_file).write_text(xyz_block)
    assert get_canonical_smiles_ob(xyz_file) == expected_smiles


@requires_openbabel()
def test_smiles_to_structure_pybel():
    # The following SMILES string failed with 100 attempts when using `smiles_to_structure_rdk`
    smiles = "O=S(=O)(O)OCC12OOC3(OO1)OC14OC(OON(S(=O)(=O)O)O1)C2C34"
    st = smiles_to_structure_pybel(smiles)
    assert st.smiles == smiles
    assert st.xyz.shape == (33, 3)
    # Check that the H atoms are added correctly
    assert len([el for el in st.elements if el == 'H']) == 7
    assert st.charge == 0
    assert st.multiplicity == 1
    assert "openbabel_time" in st.metadata

    # Raise error when given an invalid SMILES string
    invalid_smiles = "C]"
    with pytest.raises(
            ValueError,
            match=
            f"smiles_to_structure_pybel: Failed to convert '{invalid_smiles}' to format 'smi'"
    ):
        smiles_to_structure_pybel(invalid_smiles)


def test_generate_optimized_rdk_confs():
    smiles = "C#CS(=O)(=O)F"
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    for inp in [smiles, rdmol]:
        rdmol, conf_id_energies = generate_optimized_rdk_confs(
            inp, target_n_conformers=100, rmsd_threshold=0.2, max_attempts=100)
        assert rdmol.GetNumConformers() == len(conf_id_energies), (
            "Number of conformers does not match the number of energies generated"
        )
