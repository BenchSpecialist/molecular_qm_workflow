import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdMolTransforms import GetBondLength

from ..common import Structure, setup_mean_field_obj
from ..constants import HARTREE_TO_EV
from ..smiles_util import generate_optimized_rdk_confs
from ..util import get_default_logger
from .keys import DFT_ENERGY_KEY

logger = get_default_logger()

# Pre-computed F- anion energies (in Eh) for use in BDE calculations
F_ANION_ENERGIES = {
    # b3lypg/6311g*, IEF-PCM/18.5, d3bj
    "pcm_d3bj_e_tot": -99.96280459477289,
    # b3lypg/6311g*
    "gas_phase_e_tot": -99.82140880752829,
}

ENV_KEY_F_ANION_ENERGY = 'BDE_F_ANION_ENERGY'

FLUORINE_ATOM_NUMBER = 9


def _get_f_anion_energy(pyscf_options):
    """
    Get the F- anion energy based on the DFT settings.
    """
    e_f_anion = None

    # if pyscf_options._dft_level_str == "b3lypg_6311g*":
    #     if pyscf_options.solvent_method == "IEF-PCM" and \
    #         pyscf_options.solvent_eps == 18.5 and \
    #         pyscf_options.dispersion.upper() == "D3BJ":
    #         e_f_anion = F_ANION_ENERGIES["pcm_d3bj_e_tot"]

    #     if pyscf_options.solvent_method is None and pyscf_options.dispersion is None:
    #         e_f_anion = F_ANION_ENERGIES["gas_phase_e_tot"]

    if cached_f_anion_energy := os.environ.get(ENV_KEY_F_ANION_ENERGY):
        # If the environment variable is set, try use it as the F- anion energy
        try:
            e_f_anion = float(cached_f_anion_energy)
        except Exception:
            pass

    # If the F- anion energy is not set, calculate it on-the-fly
    if not isinstance(e_f_anion, float):
        # For non-default settings, calculate the F- anion energy on-the-fly
        f_anion_mol = Structure(
            elements=['F'],
            xyz=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=[FLUORINE_ATOM_NUMBER],
            charge=-1,
            multiplicity=1,
            smiles='[F-]').to_pyscf_mole(basis=pyscf_options.basis)
        mf_f_anion = setup_mean_field_obj(f_anion_mol, pyscf_options)
        mf_f_anion.kernel()
        e_f_anion = float(mf_f_anion.e_tot)
        # Cache the F- anion energy in the environment variable
        os.environ[ENV_KEY_F_ANION_ENERGY] = str(e_f_anion)
        logger.info(
            f'Calculated F- anion SPE: {e_f_anion:.6f} Eh '
            f'(cached in "{ENV_KEY_F_ANION_ENERGY}" environment variable)')

    return e_f_anion


def calc_fluoride_bond_dissociation_energy(
    st: Structure,
    pyscf_options,
    e_f_anion: float = None,
) -> Structure:
    """
    Calculate the bond dissociation energy in eV for a molecule that contains
    fluorine atoms, update the input structure with the BDE value and
    defluorined structure information.

    For anions:
    [XF-] -> X + F- (DFT settings include implicit solvent model, dispersion correction)
    For neutral molecules:
    [XF] -> X+ + F-
    """
    # Move the import inside the function to avoid circular imports
    from ..optimize import optimize_by_pyscf

    # Make sure the input structure contains the necessary information for BDE calculation
    if st.property.get(DFT_ENERGY_KEY) is None:
        error_message = (
            f"{st.smiles}(id={st.unique_id}) BDE: DFT energy not found in the input structure. "
            "Please run DFT calculation first.")
        logger.error(error_message)
        raise ValueError(error_message)

    if e_f_anion is None:
        e_f_anion = _get_f_anion_energy(pyscf_options)

    # Remove the longest fluorine bond from the molecule
    defluorined_st = get_defluorined_st(st)
    # Optimize the defluorined structure
    defluorined_st = optimize_by_pyscf(defluorined_st, options=pyscf_options)

    st.property['fluoride_bde_eV'] = (
        defluorined_st.property[DFT_ENERGY_KEY] + e_f_anion -
        st.property[DFT_ENERGY_KEY]) * HARTREE_TO_EV

    # Save info of defluorined_st to the original structure
    st.property['smiles_noF'] = defluorined_st.smiles
    st.property['charge_noF'] = defluorined_st.charge
    st.property[f'{DFT_ENERGY_KEY}_noF'] = defluorined_st.property[
        DFT_ENERGY_KEY]
    st.property['xyz_noF'] = defluorined_st.to_xyz_block()

    return st


def get_defluorined_st(st: Structure) -> Structure:
    """
    Get the defluorined structure by breaking the longest fluorine-containing
    bond in the molecule.
    """
    _msg_header = f"{st.smiles}(id={st.unique_id}) get_defluorined_st: "
    mol = Chem.MolFromXYZBlock(st.to_xyz_block())
    if mol is None:
        raise RuntimeError(_msg_header +
                           "Failed to create RDKit Mol from Structure.")

    # Assign atomic connectivity using atomic coordinates
    rdDetermineBonds.DetermineBonds(mol, charge=st.charge)

    if Chem.GetFormalCharge(mol) != st.charge:
        raise RuntimeError(
            _msg_header +
            f"Assigned charge {Chem.GetFormalCharge(mol)} does not match input charge {st.charge}."
        )

    conf = mol.GetConformer(0)

    # Find all F-containing bonds with their lengths
    f_bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
         GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in mol.GetBonds() if any(
            mol.GetAtomWithIdx(idx).GetAtomicNum() == FLUORINE_ATOM_NUMBER
            for idx in [bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()])
    ]

    if not f_bonds:
        raise RuntimeError(
            _msg_header +
            "No fluorine-containing bonds found in the RDKit mol.")

    # Get the longest bond
    begin_idx, end_idx, _ = max(f_bonds, key=lambda x: x[2])
    f_idx, other_idx = (begin_idx, end_idx) if mol.GetAtomWithIdx(
        begin_idx).GetAtomicNum() == FLUORINE_ATOM_NUMBER else (end_idx,
                                                                begin_idx)

    # Remove fluorine atom from RDKit molecule
    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveAtom(f_idx)
    defluorined_mol = editable_mol.GetMol()

    # Update the index of the former F-bonded atom if applicable
    if other_idx > f_idx:
        other_idx -= 1

    # Adjust formal charge of the bonded atom
    bonded_atom = defluorined_mol.GetAtomWithIdx(other_idx)
    bonded_atom.SetFormalCharge(bonded_atom.GetFormalCharge() + 1)
    deflu_mol_charge = Chem.GetFormalCharge(defluorined_mol)
    if deflu_mol_charge != st.charge + 1:
        raise RuntimeError(
            _msg_header +
            f"Defluorined RDKit mol has charge {deflu_mol_charge}, but expected {st.charge + 1}."
        )

    # Update atom properties like valence, hybridization after removing atoms
    defluorined_mol.UpdatePropertyCache(strict=False)

    # Sanitize
    try:
        Chem.SanitizeMol(defluorined_mol)
    except Exception as e:
        logger.error(_msg_header +
                     f"Failed to sanitize defluorined RDKit mol: {e}")

    # Generate MMFF-optimized conformers for the defluorined molecule
    defluorined_mol, conformer_energies = generate_optimized_rdk_confs(
        defluorined_mol, target_n_conformers=20)
    if len(conformer_energies) > 0:
        logger.info(
            _msg_header + f'Generated {len(conformer_energies)} conformers. '
            f'Lowest-energy conformer: {conformer_energies[0][1]:.4f} kcal/mol'
        )

        # Get lowest energy conformer coordinates
        xyz_block = Chem.MolToXYZBlock(defluorined_mol,
                                       confId=conformer_energies[0][0])
    else:
        # Fallback to the original coordinates if conformer generation fails
        xyz_block = Chem.MolToXYZBlock(defluorined_mol)

    # If defluorined_mol happens to have no conformer
    if not xyz_block:
        # Try embedding one more time without chirality constraint
        params = AllChem.ETKDGv3()
        params.enforceChirality = False
        if AllChem.EmbedMolecule(mol, params) == 0:  # embedding successful
            xyz_block = Chem.MolToXYZBlock(defluorined_mol)
        else:
            raise RuntimeError(
                _msg_header +
                "Failed to convert defluorined RDKit mol to XYZ block (no conformer)."
            )

    defluorined_st = Structure.from_xyz_block(xyz_block=xyz_block)
    defluorined_st.smiles = Chem.MolToSmiles(defluorined_mol,
                                             isomericSmiles=False,
                                             canonical=True)

    # Remove F- from the original structure, so the charge is increased by 1
    defluorined_st.charge = st.charge + 1
    # F- is closed-shell, so removing it won't change the multiplicity
    defluorined_st.multiplicity = st.multiplicity

    defluorined_st.unique_id = f'{st.unique_id}_defluorined'

    return defluorined_st
