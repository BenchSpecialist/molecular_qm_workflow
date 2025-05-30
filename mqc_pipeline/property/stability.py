import numpy as np
from ..constants import HARTREE_TO_EV


def get_homo_lumo_levels(mf) -> tuple[float, float]:
    """
    Get HOMO/LUMO energies (in eV) for electrochemical stability analysis.

    This function works for both closed-shell and open-shell systems.
    For open-shell systems, selects the highest HOMO (less negative) for
    oxidation stability and lowest LUMO (more negative) for reduction stability
    analysis.

    :param mf: PySCF mean-field object containing molecular orbital energies,
               occupations, and molecular information
    :return: Tuple of (HOMO energy in eV, LUMO energy in eV) for
             electrochemical stability calculations
    """
    # Closed-shell molecules
    if mf.mol.spin == 0:
        homo_energy = float(mf.mo_energy[mf.mo_occ > 0][-1]) * HARTREE_TO_EV
        lumo_energy = float(mf.mo_energy[mf.mo_occ == 0][0]) * HARTREE_TO_EV
        return homo_energy, lumo_energy
    else:
        # Open-shell molecules
        # Take the higher HOMO (less negative) for oxidation stability/oxidation potential
        homo_energy = max(mf.mo_energy[0][mf.mo_occ[0] > 0][-1],
                          mf.mo_energy[1][mf.mo_occ[1] > 0][-1])

        # Take the lower LUMO (more negative) for reduction stability/reduction potential
        lumo_energy = min(mf.mo_energy[0][mf.mo_occ[0] == 0][0],
                          mf.mo_energy[1][mf.mo_occ[1] == 0][0])
        return float(homo_energy) * HARTREE_TO_EV, float(
            lumo_energy) * HARTREE_TO_EV


def _get_open_shell_averaged_levels(mf) -> tuple[float, float]:
    """
    Get average alpha/beta spin orbital levels for open-shell systems.

    This approach calculates HOMO/LUMO energies by averaging the corresponding
    alpha and beta spin orbital energies. This method is particularly relevant
    for electrochemical applications where the average electronic environment
    better represents the system's redox behavior.

    :param mf: PySCF mean-field object containing orbital energies and occupations
    :return: Tuple of (average HOMO energy in eV, average LUMO energy in eV).
             Returns (None, None) for closed-shell systems where spin separation
             is not applicable
    """
    if mf.mo_energy.ndim == 1:
        return None, None

    # Average orbital energies
    alpha_energies = mf.mo_energy[0]
    beta_energies = mf.mo_energy[1]

    # Find occupied orbitals in both spins
    alpha_homo = np.where(mf.mo_occ[0] > 0)[0][-1]
    beta_homo = np.where(mf.mo_occ[1] > 0)[0][-1]

    # Average HOMO energy
    homo_avg = 0.5 * (alpha_energies[alpha_homo] + beta_energies[beta_homo])

    # For LUMO, take the lowest virtual orbital
    alpha_lumo = np.where(mf.mo_occ[0] == 0)[0][0]
    beta_lumo = np.where(mf.mo_occ[1] == 0)[0][0]
    lumo_avg = 0.5 * (alpha_energies[alpha_lumo] + beta_energies[beta_lumo])

    return float(homo_avg) * HARTREE_TO_EV, float(lumo_avg) * HARTREE_TO_EV


def _get_open_shell_effective_levels(mf,
                                     homo_threshold=1.0,
                                     lumo_threshold=1.0
                                     ) -> tuple[float, float]:
    """
    Identifies orbitals with significant electron density regardless of spin
    for open-shell systems.

    This function calculates HOMO/LUMO energies based on total orbital occupations
    (alpha + beta) rather than individual spin channels. It's useful for systems
    where fractional occupations or spin contamination may affect the traditional
    HOMO/LUMO identification.

    :param mf: PySCF mean-field object containing orbital energies and occupations
    :param homo_threshold: Minimum total occupation for an orbital to be considered
                          as HOMO candidate (default: 1.0)
    :param lumo_threshold: Maximum total occupation for an orbital to be considered
                          as LUMO candidate (default: 1.0)
    :return: Tuple of (HOMO energy in eV, LUMO energy in eV). Returns (None, None)
             for closed-shell systems or if no suitable orbitals are found
    """
    if mf.mo_energy.ndim == 1:
        return None, None
    # Calculate fractional occupations
    alpha_occ = mf.mo_occ[0]
    beta_occ = mf.mo_occ[1]
    total_occ = alpha_occ + beta_occ

    homo_candidates = np.where(total_occ >= homo_threshold)[0]
    lumo_candidates = np.where(total_occ < lumo_threshold)[0]

    homo_energy, lumo_energy = None, None
    if len(homo_candidates) > 0:
        homo_idx = homo_candidates[-1]
        homo_energy = min(mf.mo_energy[0][homo_idx], mf.mo_energy[1][homo_idx])
        homo_energy = float(homo_energy) * HARTREE_TO_EV

    if len(lumo_candidates) > 0:
        lumo_idx = lumo_candidates[0]
        lumo_energy = max(mf.mo_energy[0][lumo_idx], mf.mo_energy[1][lumo_idx])
        lumo_energy = float(lumo_energy) * HARTREE_TO_EV

    return homo_energy, lumo_energy
