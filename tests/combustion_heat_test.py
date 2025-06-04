from collections import namedtuple
import numpy as np
import pytest

from mqc_pipeline.property.combustion_heat import calc_combustion_heat
from mqc_pipeline.constants import HARTREE_TO_EV
from mqc_pipeline.common import Structure
from mqc_pipeline.smiles_util import smiles_to_structure_rdk

from .conftest import requires_openbabel

TestCase = namedtuple(
    'TestCase',
    ['smiles', 'mol_heat', 'expected_combustion_heat', 'reaction_str'])

test_cases = [
    # Nonflammable systems
    TestCase('OC(O)(F)F', 0, 0, ''),
    TestCase('OO', 0, 0, ''),
    # Flammable systems
    TestCase('CO', 0, -3158.019652569623,
             '1.00 * CO + 1.50 * O2 -> 2.00 * H2O + 1.00 * CO2'),
    TestCase('[C-]#[O+]', 0, -3088.2150793075243,
             '1.00 * [C-]#[O+] + 0.50 * O2 -> 1.00 * CO2'),
    TestCase(
        'CN(C)S(=O)(=O)F', 0, -21327.63350042753,
        '1.00 * CN(C)S(=O)(=O)F + 3.25 * O2 -> 2.50 * H2O + 2.00 * CO2 + 0.50 * N2 + 1.00 * HF + 1.00 * SO2'
    ),
    TestCase(
        'O=S(=O)(F)CCCCCF', -944.9859370316980 * HARTREE_TO_EV,
        -38.37295288985479,
        '1.00 * O=S(=O)(F)CCCCCF + 7.00 * O2 -> 4.00 * H2O + 5.00 * CO2 + 2.00 * HF + 1.00 * SO2'
    ),
    TestCase(
        'FC(Cl)OP(=S)(OC)F', 0, -44338.306357703754,
        '1.00 * FC(Cl)OP(=S)(OC)F + 3.50 * O2 -> 0.50 * H2O + 2.00 * CO2 + 2.00 * HF + 0.50 * P2O5 + 1.00 * SO2 + 1.00 * HCl'
    ),
    TestCase(
        'CN(C)C(=O)OP(=S)(F)Cl', 0, -44190.87101069842,
        '1.00 * CN(C)C(=O)OP(=S)(F)Cl + 5.25 * O2 -> 2.00 * H2O + 3.00 * CO2 + 0.50 * N2 + 1.00 * HF + 0.50 * P2O5 + 1.00 * SO2 + 1.00 * HCl'
    ),
    TestCase(
        'FC[Si](Cl)(Cl)OP(=S)(N)C=O', 0, -63535.52364373139,
        '1.00 * FC[Si](Cl)(Cl)OP(=S)(N)C=O + 4.75 * O2 -> 1.00 * H2O + 2.00 * CO2 + 0.50 * N2 + 1.00 * HF + 1.00 * SiO2 + 0.50 * P2O5 + 1.00 * SO2 + 2.00 * HCl'
    )
]


@pytest.mark.parametrize("test_case",
                         test_cases,
                         ids=[case.smiles for case in test_cases])
def test_calc_combustion_heat(test_case):
    """
    Test combustion heat calculation starting with SMILES strings.
    """
    combustion_heat, reaction = calc_combustion_heat(test_case.smiles,
                                                     test_case.mol_heat)
    assert np.isclose(combustion_heat, test_case.expected_combustion_heat, atol=1e-06), \
        f"Expected {test_case.expected_combustion_heat} but got {combustion_heat} for SMILES: {test_case.smiles}"
    assert reaction == test_case.reaction_str, \
        f"Expected reaction string '{test_case.reaction_str}' but got '{reaction}' for SMILES: {test_case.smiles}"


@requires_openbabel()
def test_calc_combustion_heat_st():
    """
    Test combustion heat calculation starting with a Structure object.
    """
    coh4_xyz = """6
CO
C          -0.34832       -0.03472       -0.04218
O           1.02245       -0.38610       -0.14632
H          -0.68493        0.04087        1.00077
H          -0.58053        0.90876       -0.55474
H          -0.91464       -0.83162       -0.52485
H           1.54581        0.30417        0.27394
"""
    st = Structure.from_xyz_block(xyz_block=coh4_xyz)
    e_tot = -115.746534 * 27.2114  # in eV
    ref_heat = -8.394417282022914

    combustion_heat, reaction = calc_combustion_heat(st, mol_heat=e_tot)
    assert np.isclose(combustion_heat, ref_heat, atol=1e-06)
    assert reaction == '1.00 * CO + 1.50 * O2 -> 2.00 * H2O + 1.00 * CO2'


def test_unsupported_elements():
    """
    Test that unsupported elements raise a ValueError.
    """
    smiles = "C[Si](C)(C)O[Al](O[Si](C)(C)C)O[Si](C)(C)C"
    with pytest.raises(
            ValueError,
            match='Unsupported element Al for combustion heat calculation.'):
        calc_combustion_heat(smiles, mol_heat=0)

    st = smiles_to_structure_rdk(smiles)
    with pytest.raises(
            ValueError,
            match='Unsupported element Al for combustion heat calculation.'):
        calc_combustion_heat(st, mol_heat=0)


def test_unsupported_dft_level():
    dft_level_unsupported = 'wb97mv_def2tzvppd'
    with pytest.raises(
            NotImplementedError,
            match=
            f'Combustion heat calculation for DFT level "{dft_level_unsupported}" is not available.'
    ):
        calc_combustion_heat("C", mol_heat=0, dft_level=dft_level_unsupported)
