import time
import numpy as np

from ..util import logger

try:
    from gpu4pyscf.dft import rks, uks
    from gpu4pyscf.qmmm import chelpg
    _USE_GPU = True
except (ImportError, AttributeError):
    from pyscf.dft import rks, uks
    _USE_GPU = False

from ..common import Structure
from ..constants import HARTREE_TO_EV
from ..settings import PySCFOption, ESPGridsOption

from .esp import generate_esp_grids, get_esp_range
from .keys import (DFT_ENERGY_KEY, HOMO_KEY, LUMO_KEY, ESP_MIN_KEY,
                   ESP_MAX_KEY, DIPOLE_X_KEY, DIPOLE_Y_KEY, DIPOLE_Z_KEY,
                   DFT_FORCES_KEY, CHELPG_CHARGE_KEY)


def _is_scf_done(mf_obj) -> bool:
    """
    Check if the SCF calculation is done.

    :param mf_obj: PySCF mean-field object.
    """
    return mf_obj.e_tot != 0


def get_quadrupole_moment(
        mf_obj, rdm1) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the quadrupole moment of a molecule.

    :param mol: PySCF Mole object.
    :param rdm1: One-body reduced density matrix.

    :return: Quadrupole moment expressed in 6 components (Qxx, Qyy, Qzz, Qxy, Qxz, Qyz).
    """
    if not _is_scf_done(mf_obj):
        raise RuntimeError(
            "Quadrupole: No SCF calculation performed. Please run SCF first.")

    # PySCF returns the traceless quadrupole moment as a full symmetric matrix;
    # only upper or lower triangle is needed
    quad_moment = mf_obj.quad_moment(unit='DebyeAngstrom', dm=rdm1)
    qxx, qyy, qzz = [float(val) for val in np.diag(quad_moment)]
    qxy = float(quad_moment[0, 1])
    qxz = float(quad_moment[0, 2])
    qyz = float(quad_moment[1, 2])

    return qxx, qyy, qzz, qxy, qxz, qyz


def get_default_properties(st: Structure, mf, rdm1) -> Structure:
    """
    Get default properties for the given structure. This function wraps calculations
    of shared properties among neutral solvent and ions, including:
    Total electronic energy (Hartree), HOMO (eV), LUMO (eV), dipole (Debye).

    :param st: Structure object containing the molecule information.
    :param mf: PySCF mean-field object.
    :param rdm1: np.ndarray represents one-body density matrix

    :return: Structure object with populated `property` attribute.
    """
    if not _is_scf_done(mf):
        raise RuntimeError(
            "Default Properties: No SCF calculation performed. Please run SCF first."
        )

    # Get the total SCF energy
    st.property[DFT_ENERGY_KEY] = float(mf.e_tot)

    # Get HOMO and LUMO energies (neutral species only) in eV.
    st.property[HOMO_KEY] = float(
        mf.mo_energy[mf.mo_occ > 0][-1]) * HARTREE_TO_EV
    st.property[LUMO_KEY] = float(
        mf.mo_energy[mf.mo_occ == 0][0]) * HARTREE_TO_EV

    # Get dipole
    st.property[DIPOLE_X_KEY], st.property[DIPOLE_Y_KEY], st.property[
        DIPOLE_Z_KEY] = mf.dip_moment(unit='Debye', dm=rdm1)
    return st


def get_properties_neutral(st: Structure,
                           pyscf_options: PySCFOption,
                           esp_options: ESPGridsOption,
                           return_gradient: bool = False,
                           return_chelpg_chg: bool = True) -> Structure:
    """
    Compute properties for the given structure (neutral molecule) using PySCF.

    :param st: Structure object containing the molecule information.
    :param pyscf_options: PySCFOption object with relevant parameters to set up PySCF.
    :param return_gradient: Whether to run gradient calculations. This needs to be
                            true if the input structure is not optimized by PySCF,
                            as the DFT gradients are not available from the geometry
                            optimization.

    :return: Structure object with populated `property` attribute.
    """
    t_start = time.perf_counter()

    mol = st.to_pyscf_mole(basis=pyscf_options.basis)

    # Setup Kohn-Sham DFT object
    if st.multiplicity == 1:  # closed-shell
        mf = rks.RKS(mol, xc=pyscf_options.dft_functional).density_fit()
    else:
        mf = uks.UKS(mol, xc=pyscf_options.dft_functional).density_fit()

    mf.max_cycle = pyscf_options.max_scf_cycle
    mf.conv_tol = pyscf_options.scf_conv_tol
    mf.grids.level = pyscf_options.grids_level

    # Run SCF calculation
    mf.kernel()

    # Compute one-body reduced density matrix, which is used in:
    # - electrostatic potential (ESP) calculations
    # - dipole, [quadrupole] calculations
    rdm1 = mf.make_rdm1()

    # Get default properties: e_tot, HOMO, LUMO, dipole
    st = get_default_properties(st, mf, rdm1)

    if return_gradient:
        gradients_arr = mf.Gradients().kernel()
        st.save_gradients(gradients_arr, prop_key=DFT_FORCES_KEY)

    # ESP and CHELPG charges calculations can only run on GPUs
    if _USE_GPU:
        # Generate grids for ESP calculations
        grids = generate_esp_grids(mol,
                                   rcut=esp_options.solvent_accessible_region,
                                   space=esp_options.grid_spacing,
                                   solvent_probe=esp_options.probe_depth)
        st.property[ESP_MIN_KEY], st.property[ESP_MAX_KEY] = get_esp_range(
            mol, grids, one_rdm=rdm1)

        if return_chelpg_chg:
            # Evaluate CHELPG charges and transfers data from GPU (cupy) to CPU (numpy)
            # CHELPG method fits atomic charges to reproduce ESP at a number of points
            # around the molecule.
            chelpg_charges = chelpg.eval_chelpg_layer_gpu(mf).get()
            st.atom_property[CHELPG_CHARGE_KEY] = chelpg_charges
    else:
        logger.warning(
            "CHELPG charges and ESP calculations are not available without GPU."
        )

    # PySCF is not expected to change the order of atoms, but we update it just in case
    st.elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    st.atomic_numbers = [mol.atom_charge(i) for i in range(mol.natm)]

    st.metadata['dft_prop_calc_duration'] = time.perf_counter() - t_start
    return st
