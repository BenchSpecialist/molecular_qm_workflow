import numpy as np
import pytest
from mqc_pipeline.property.combustion_heat import calc_combustion_heat
from mqc_pipeline.constants import HARTREE_TO_EV


@pytest.mark.parametrize(
    "smiles, mol_heat, expected_combustion_heat, reaction_str",
    [
        # Nonflammable systems
        ('OC(O)(F)F', 0, 0, ''),
        ('OO', 0, 0, ''),
        # Flammable systems
        ('CO', 0, -5524.848919127996,
         '1.00 * CO + 1.50 * O2 -> 2.00 * H2O + 1.00 * CO2'),
        ('[C-]#[O+]', 0, -5455.044310596723,
         '1.00 * [C-]#[O+] + 0.50 * O2 -> 1.00 * CO2'),
        ('CN(C)S(=O)(=O)F', 0, -26061.299620746897,
         '1.00 * CN(C)S(=O)(=O)F + 3.25 * O2 -> 2.50 * H2O + 2.00 * CO2 + 0.50 * N2 + 1.00 * HF + 1.00 * SO2'
         ),
        ('O=S(=O)(F)CCCCCF', -944.9859370316980 * HARTREE_TO_EV,
         -11872.511323547747,
         '1.00 * O=S(=O)(F)CCCCCF + 7.00 * O2 -> 4.00 * H2O + 5.00 * CO2 + 2.00 * HF + 1.00 * SO2'
         ),
        ('FC(Cl)OP(=S)(OC)F', 0, -49071.98406824183,
         '1.00 * FC(Cl)OP(=S)(OC)F + 3.50 * O2 -> 0.50 * H2O + 2.00 * CO2 + 2.00 * HF + 0.50 * P2O5 + 1.00 * SO2 + 1.00 * HCl'
         ),
        ('CN(C)C(=O)OP(=S)(F)Cl', 0, -51291.37631704636,
         '1.00 * CN(C)C(=O)OP(=S)(F)Cl + 5.25 * O2 -> 2.00 * H2O + 3.00 * CO2 + 0.50 * N2 + 1.00 * HF + 0.50 * P2O5 + 1.00 * SO2 + 1.00 * HCl'
         ),
        ('FC[Si](Cl)(Cl)OP(=S)(N)C=O', 0, -68269.21110839846,
         '1.00 * FC[Si](Cl)(Cl)OP(=S)(N)C=O + 4.75 * O2 -> 1.00 * H2O + 2.00 * CO2 + 0.50 * N2 + 1.00 * HF + 1.00 * SiO2 + 0.50 * P2O5 + 1.00 * SO2 + 2.00 * HCl'
         )
    ])
def test_calc_combustion_heat(smiles, mol_heat, expected_combustion_heat,
                              reaction_str):
    combustion_heat, reaction = calc_combustion_heat(smiles, mol_heat)
    assert np.allclose(combustion_heat, expected_combustion_heat, atol=1e-09), \
        f"Expected {expected_combustion_heat} but got {combustion_heat} for SMILES: {smiles}"
    assert reaction == reaction_str, \
        f"Expected reaction string '{reaction_str}' but got '{reaction}' for SMILES: {smiles}"
