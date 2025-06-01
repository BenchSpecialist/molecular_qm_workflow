import numpy as np
from dataclasses import dataclass, field
from rdkit import Chem

from ..constants import HARTREE_TO_EV
from ..common import Structure
from ..smiles_util import get_canonical_smiles_ob


@dataclass(frozen=True)
class ElementCombustionData:
    """Data for element combustion properties."""
    product: str
    o2_consumption: float
    product_ratio: float
    product_heat_hartree: float
    product_heat_ev: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'product_heat_ev',
                           self.product_heat_hartree * HARTREE_TO_EV)


ELEMENT_DATA = {
    'H': ElementCombustionData('H2O', 0.5, 0.5, -76.43394949999472),
    'Li': ElementCombustionData('Li2O', 0.5, 0.5, np.nan),
    'Be': ElementCombustionData('BeO', 1, 1, np.nan),
    'B': ElementCombustionData('B2O3', 1.5, 0.5, -275.6203936495455),
    'C': ElementCombustionData('CO2', 2, 1, -275.6203936495455),
    'N': ElementCombustionData('N2', 0, 0.5, -109.555929820176),
    'O': ElementCombustionData('O2', -1, 0, -150.3026282483459),
    'F': ElementCombustionData('HF', 0, 1, -100.45856381466965),
    'Na': ElementCombustionData('Na2O', 0.5, 1, np.nan),
    'Si': ElementCombustionData('SiO2', 2, 1, -440.0005335019524),
    # e_total of P2O5 is obtained by dividing e_total of P4O10 by 2
    'P': ElementCombustionData('P2O5', 2.5, 0.5, -1059.1269704500623),
    'S': ElementCombustionData('SO2', 2, 1, -548.6559640248399),
    'Cl': ElementCombustionData('HCl', 0, 1, -460.8261007532856),
    'K': ElementCombustionData('K2O', 0.5, 1, np.nan),
    'Br': ElementCombustionData('HBr', 0, 1, -2574.7474248325434),
    'I': ElementCombustionData('HI', 0, 1, -6920.127908312316)
}


def calc_combustion_heat(smiles_or_st: str | Structure,
                         mol_heat: float = 0) -> tuple[float, str]:
    """
    Calculate the heat of combustion for a molecule given its SMILES string or
    a Structure object and DFT energy.

    :param smiles_or_st: SMILES string of the molecule or a Structure object
    :param mol_heat: Energy of the molecule from DFT in eV
    :return: A tuple containing the heat of combustion in eV and the reaction string
    """
    ele_count = {key: 0 for key in ELEMENT_DATA.keys()}
    if isinstance(smiles_or_st, str):
        smiles = smiles_or_st
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            raise ValueError(f'Invalid SMILES: {smiles}')

        # Add implicit hydrogens so H atoms are counted
        mol = Chem.AddHs(mol)

        # Count elements
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            if element in ele_count:
                ele_count[element] += 1
            else:
                raise ValueError(
                    f'Unsupported element {element} for combustion heat calculation.'
                )

    if isinstance(smiles_or_st, Structure):
        st = smiles_or_st
        for element in st.elements:
            if element in ele_count:
                ele_count[element] += 1
            else:
                raise ValueError(
                    f'Unsupported element {element} for combustion heat calculation.'
                )
        if not st.smiles or st.smiles.isspace():
            # Convert Structure to H-contained SMILES if not provided
            smiles = get_canonical_smiles_ob(st.to_xyz_block())
        else:
            smiles = st.smiles

    # Check if nonflammable due to halogen content
    halogen_count = sum(ele_count[x] for x in ['F', 'Cl', 'Br', 'I'])
    if halogen_count >= ele_count['H'] and ele_count['H'] != 0:
        print('Our model predict this material as nonflammable')
        return 0, ''

    # H left after halogen formation
    ele_count['H'] -= halogen_count

    # Calculate total oxygen consumption
    total_oxy_consump = sum(count * ELEMENT_DATA[element].o2_consumption
                            for element, count in ele_count.items())

    if total_oxy_consump <= 0:
        print('Our model predict this material as nonflammable')
        return 0, ''

    # Calculate heat of products using the pre-converted ev field
    product_heat_sum = sum(
        count * ELEMENT_DATA[element].product_ratio *
        ELEMENT_DATA[element].product_heat_ev
        for element, count in ele_count.items()
        if not np.isnan(ELEMENT_DATA[element].product_heat_ev))

    # Calculate heat of reactants
    reactant_heat_sum = mol_heat + (total_oxy_consump /
                                    2.0) * ELEMENT_DATA['O'].product_heat_ev
    combustion_heat = product_heat_sum - reactant_heat_sum

    # Build reaction string
    reaction = f"1.00 * {smiles} + {total_oxy_consump / 2.0:.2f} * O2 ->"

    _first_product = True
    for element, count in ele_count.items():
        if (product_amount := count * ELEMENT_DATA[element].product_ratio) > 0:
            prefix = '' if _first_product else ' +'
            reaction += f"{prefix} {product_amount:.2f} * {ELEMENT_DATA[element].product}"
            _first_product = False

    return combustion_heat, reaction
