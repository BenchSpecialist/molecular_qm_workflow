import polars
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from rdkit import Chem

from ..constants import HARTREE_TO_EV
from ..common import Structure
from ..smiles_util import get_canonical_smiles_ob
from ..util import get_default_logger

_COMBUSTION_CSV = Path(__file__).parent / "ElementCombustionData.csv"
_COMBUSTION_DF = polars.read_csv(_COMBUSTION_CSV)

logger = get_default_logger()


@dataclass(frozen=True)
class ElementCombustionData:
    """Data for element combustion properties."""
    product: str
    o2_consumption: float
    product_ratio: float
    product_e_tot_hartree: float
    product_H_tot_298K_ev: float = np.nan
    product_e_tot_ev: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'product_e_tot_ev',
                           self.product_e_tot_hartree * HARTREE_TO_EV)

    @classmethod
    def from_dict(cls,
                  data_dict: dict,
                  dft_level: str = 'b3lypg_6311g*') -> 'ElementCombustionData':
        # Extract required fields from dictionary
        return cls(
            product=data_dict['product'],
            o2_consumption=data_dict['o2_consumption'],
            product_ratio=data_dict['product_ratio'],
            product_e_tot_hartree=data_dict[f'{dft_level}_e_tot_hartree'],
            product_H_tot_298K_ev=data_dict.get(f'{dft_level}_H_tot_298K_ev',
                                                np.nan))


ELEMENT_DATA = {
    row["element"]: ElementCombustionData.from_dict(row)
    for row in _COMBUSTION_DF.iter_rows(named=True)
}


def calc_combustion_heat(
        smiles_or_st: str | Structure,
        mol_heat: float = 0.0,
        dft_level: str = 'b3lypg_6311g*') -> tuple[float, str]:
    """
    Calculate the heat of combustion for a molecule given its SMILES string or
    a Structure object and DFT energy.

    :param smiles_or_st: SMILES string of the molecule or a Structure object
    :param mol_heat: Energy of the molecule from DFT in eV
    :param dft_level: String indicating the DFT level of theory used for the calculation;
                      format: f'{functional}_{basis_set}' where basis_set string has no '-'.
    :return: A tuple containing the heat of combustion in eV and the reaction string
    """
    if not (keys :=
            [key for key in _COMBUSTION_DF.columns if dft_level in key]):
        raise NotImplementedError(
            f'Combustion heat calculation for DFT level "{dft_level}" is not available.'
        )

    ele_count = {ele: 0 for ele in _COMBUSTION_DF['element'].to_list()}
    if isinstance(smiles_or_st, str):
        smiles = smiles_or_st
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            raise ValueError(f'Invalid SMILES: {smiles}')

        # Add implicit hydrogens so H atoms are counted
        mol = Chem.AddHs(mol)

        # Count elements
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            if element not in ele_count:
                logger.error(
                    f'Unsupported element {element} for combustion heat calculation.'
                )
                return 0.0, ''
            ele_count[element] += 1

    if isinstance(smiles_or_st, Structure):
        st = smiles_or_st
        for element in st.elements:
            if element not in ele_count:
                logger.error(
                    f'Unsupported element {element} for combustion heat calculation.'
                )
                return 0.0, ''
            ele_count[element] += 1

        if not st.smiles or st.smiles.isspace():
            # Convert Structure to H-contained SMILES if not provided
            smiles = get_canonical_smiles_ob(st.to_xyz_block())
        else:
            smiles = st.smiles

    # Check if nonflammable due to halogen content
    halogen_count = sum(ele_count[x] for x in ['F', 'Cl', 'Br', 'I'])
    if halogen_count >= ele_count['H'] and ele_count['H'] != 0:
        logger.info(f'{smiles}: nonflammable material due to halogen.')
        return 0.0, ''

    # H left after halogen formation
    ele_count['H'] -= halogen_count

    # Calculate total oxygen consumption
    total_oxy_consump = sum(count * ELEMENT_DATA[element].o2_consumption
                            for element, count in ele_count.items())

    if total_oxy_consump <= 0:
        logger.info(f'{smiles}: nonflammable material.')
        return 0.0, ''

    # Calculate heat of products using the pre-converted ev field
    product_heat_sum = sum(
        count * ELEMENT_DATA[element].product_ratio *
        ELEMENT_DATA[element].product_e_tot_ev
        for element, count in ele_count.items()
        if not np.isnan(ELEMENT_DATA[element].product_e_tot_ev))

    # Calculate heat of reactants
    reactant_heat_sum = mol_heat + (total_oxy_consump /
                                    2.0) * ELEMENT_DATA['O'].product_e_tot_ev
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
