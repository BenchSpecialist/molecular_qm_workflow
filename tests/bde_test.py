from unittest.mock import patch
import pytest

from mqc_pipeline.property import bde
from mqc_pipeline.property.keys import DFT_ENERGY_KEY
from mqc_pipeline.common import Structure
from mqc_pipeline.settings import PySCFOption


@pytest.fixture(scope="function")
def test_st():
    """
    Create a test structure from the provided XYZ block.
    """
    xyz_block = """7
O=C(F)N=S(F)F
O -2.061840759389956 0.8169631724380221 -0.6440545161100142
C -1.322266671294358 0.06637545080252248 0.08067009911339568
F -1.7431105619973024 -1.1914517893983803 0.36247982548821184
N -0.08789557786613517 0.5967910313027855 0.5555836530077778
S 1.2921788776178942 0.004666086108818865 0.2066480685348283
F 1.335995173006138 -1.3146069135917382 -0.7699585918554823
F 2.5869395199237206 1.0212629623379625 0.20863146182127693
"""
    st = Structure.from_xyz_block(xyz_block)
    st.smiles = "O=C(F)N=S(F)F"
    st.unique_id = "123456"
    st.property[DFT_ENERGY_KEY] = -865.8958017126204
    return st


def test_get_defluorined_st(test_st):
    defluorined_st = bde.get_defluorined_st(test_st)

    # Removing one F- from the neutral molecule leaves a cation
    assert defluorined_st.charge == 1
    assert defluorined_st.multiplicity == 1
    assert len(defluorined_st.elements) == len(test_st.elements) - 1
    assert defluorined_st.smiles == '[O-]C(F)=N[S+2]F'


@patch('mqc_pipeline.optimize.optimize_by_pyscf')
def test_calc_fluoride_bond_dissociation_energy(mock_optimize, test_st):
    expected_bde_in_ev = 4.122

    defluorined_st = bde.get_defluorined_st(test_st)
    defluorined_st.property[DFT_ENERGY_KEY] = -765.7816053641884
    mock_optimize.return_value = defluorined_st

    pyscf_options = PySCFOption()
    st = bde.calc_fluoride_bond_dissociation_energy(
        test_st,
        e_f_anion=bde.F_ANION_ENERGIES['pcm_d3bj_e_tot'],
        pyscf_options=pyscf_options)
    assert st.property['fluoride_bde_eV'] == pytest.approx(expected_bde_in_ev,
                                                           abs=1e-2)
