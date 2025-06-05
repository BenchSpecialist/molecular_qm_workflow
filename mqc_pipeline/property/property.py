import os
import time
import json
import numpy as np
from pathlib import Path
from functools import lru_cache

from ..util import get_default_logger, timeit

from ..common import Structure, setup_mean_field_obj
from ..settings import PySCFOption, ESPGridsOption

from .stability import get_homo_lumo_levels
from .esp import generate_esp_grids, get_esp_range
from .volume import get_vdw_volume
from .combustion_heat import calc_combustion_heat
from .keys import (DFT_ENERGY_KEY, HOMO_KEY, LUMO_KEY, ESP_MIN_KEY,
                   ESP_MAX_KEY, DIPOLE_X_KEY, DIPOLE_Y_KEY, DIPOLE_Z_KEY,
                   DFT_FORCES_KEY, CHELPG_CHARGE_KEY)

logger = get_default_logger()


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@lru_cache(maxsize=1)
def _import_chelpg():
    try:
        from gpu4pyscf.qmmm import chelpg
        return chelpg
    except (ImportError, AttributeError):
        logger.error('CHELPG error: gpu4pyscf.qmmm.chelpg is not available.')
        return None


def _is_scf_done(mf_obj) -> bool:
    """
    Check if the SCF calculation is done.

    :param mf_obj: PySCF mean-field object.
    """
    return mf_obj.e_tot != 0


@lru_cache(maxsize=1)
def _import_thermo():
    """Import thermo module once and cache it."""
    from pyscf.hessian import thermo
    return thermo


def get_thermo_info(mf_obj, unique_id: str = ''):
    """
    Compute Hessian and return frequency and thermochemical properties
    """
    thermo = _import_thermo()
    if not _is_scf_done(mf_obj):
        raise RuntimeError(
            "Freq: No SCF calculation performed. Please run SCF first.")

    hessian = mf_obj.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mf_obj.mol,
                                         hessian,
                                         imaginary_freq=False)
    if freq_info['freq_wavenumber'] is not None and (
            freq_info['freq_wavenumber'] < 0).any():
        freq_info_file = Path(f"negative_freqs_{unique_id}.json").resolve()
        with open(freq_info_file, 'w') as f:
            json.dump(freq_info, f, cls=NumpyEncoder)
        logger.warning(
            "Negative frequencies detected; structure may not be a equilibrium geometry. "
            f"Save freq information to file: {freq_info_file}.")
        return

    # Only calculate thermochemical properties for equilibrium geometry
    thermo_info = thermo.thermo(mf_obj,
                                freq_info['freq_au'],
                                temperature=298.15,
                                pressure=101325)
    thermo_info.update(freq_info)
    return thermo_info


@lru_cache(maxsize=1)
def _import_polarizability():
    from gpu4pyscf.properties import polarizability
    return polarizability


def get_isotropic_polarizability(mf_obj) -> float:
    if not _is_scf_done(mf_obj):
        raise RuntimeError(
            "Polarizability: No SCF calculation performed. Please run SCF first."
        )

    polarizability = _import_polarizability()
    # Get the full 3x3 polarizability tensor
    alpha_tensor = polarizability.eval_polarizability(mf_obj)
    # Isotropic averaging
    alpha_iso = float(np.trace(alpha_tensor) / 3.0)
    return alpha_iso


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
    - Total electronic energy (Hartree)
    - HOMO (eV), LUMO (eV)
    - Dipole (Debye)

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

    # Get HOMO and LUMO energies in eV.
    st.property[HOMO_KEY], st.property[LUMO_KEY] = get_homo_lumo_levels(mf)

    # Get dipole
    st.property[DIPOLE_X_KEY], st.property[DIPOLE_Y_KEY], st.property[
        DIPOLE_Z_KEY] = mf.dip_moment(unit='Debye', dm=rdm1)
    return st


def get_properties_main(st: Structure,
                        pyscf_options: PySCFOption,
                        esp_options: ESPGridsOption,
                        return_esp_range: bool = True,
                        return_combustion_heat: bool = True,
                        return_gradient: bool = False,
                        return_chelpg_chg: bool = True,
                        return_freq: bool = False,
                        return_volume: bool = False,
                        return_polarizability: bool = False,
                        return_quadrupole: bool = False) -> Structure:
    """
    Compute properties for the given structure (neutral molecule) using PySCF.

    :param st: Structure object containing the molecule information.
    :param pyscf_options: PySCFOption object with relevant parameters to set up PySCF.
    :param return_chelpg_chg: Whether to calculate CHELPG partial charges.
    :param return_gradient: Whether to run gradient calculations. This needs to be
                            true if the input structure is not optimized by PySCF,
                            as the DFT gradients are not available from the geometry
                            optimization.
    :param return_freq: Whether to run frequency calculations.
    :param return_quadrupole: Whether to calculate the quadrupole moment.

    :return: Structure object with populated `property` attribute.
    """
    t_start = time.perf_counter()

    mol = st.to_pyscf_mole(basis=pyscf_options.basis)

    # Setup Kohn-Sham DFT object
    mf = setup_mean_field_obj(mol, pyscf_options)

    # Run SCF calculation
    mf.kernel()

    # Compute one-body reduced density matrix, which is used in:
    # - electrostatic potential (ESP) calculations
    # - dipole, [quadrupole] calculations
    rdm1 = mf.make_rdm1()

    # Get default properties: e_tot, HOMO, LUMO, dipole
    st = get_default_properties(st, mf, rdm1)

    # ESP calculations can only run on GPUs
    if return_esp_range:
        # Generate grids for ESP calculations
        grids = generate_esp_grids(mol,
                                   rcut=esp_options.solvent_accessible_region,
                                   space=esp_options.grid_spacing,
                                   solvent_probe=esp_options.probe_depth)
        esp_range, st.metadata['dft_esp_time'] = timeit(
            get_esp_range, mol=mol, grids=grids, one_rdm=rdm1) # yapf:disable
        st.property[ESP_MIN_KEY], st.property[ESP_MAX_KEY] = esp_range

    # CHELPG charges calculations are GPU-only
    if return_chelpg_chg:
        chelpg = _import_chelpg()
        # Evaluate CHELPG charges: this method fits atomic charges to reproduce
        # ESP at a number of points around the molecule.
        st.atom_property[CHELPG_CHARGE_KEY], st.metadata[
            'dft_chelpg_time'] = timeit(
                lambda mf: chelpg.eval_chelpg_layer_gpu(mf).get(), mf=mf)

    # Currently, the total electronic energy is used to approximate the enthalpy
    if return_combustion_heat:
        try:
            st.property['combustion_heat_eV'], _ = calc_combustion_heat(
                st.smiles, mol_heat=st.property[DFT_ENERGY_KEY])
        except Exception as e:
            logger.error(
                f"{st.smiles} (id={st.unique_id}): Failed to calculate combustion heat: {str(e)}"
            )
            st.property['combustion_heat_eV'] = None

    if return_gradient:
        gradients_arr, st.metadata['dft_gradient_time'] = timeit(
            lambda mf: mf.Gradients().kernel(), mf=mf)
        st.save_gradients(gradients_arr, prop_key=DFT_FORCES_KEY)

    if return_freq:
        thermo_info, st.metadata['dft_hessian_time'] = timeit(
            get_thermo_info, mf_obj=mf, unique_id=st.unique_id)
        if thermo_info:
            # Save ZPE, E_0K, E_tot, H_tot, G_tot, Cv_tot
            st.save_thermo_info(thermo_info)

    if return_quadrupole:
        st.property['quadrupole_debyeAngstrom'] = get_quadrupole_moment(
            mf, rdm1)

    if return_volume:
        st.property['vdw_volume_angstrom3'], st.metadata[
            'dft_vdw_volume_time'] = timeit(get_vdw_volume,
                                            coords_angstrom=mol.atom_coords(),
                                            atomic_numbers=mol.atom_charges(),
                                            use_gpu=True)

    if return_polarizability and st.multiplicity == 1:
        # Backend function only supports closed-shell systems
        try:
            st.property['alpha_iso_au'], st.metadata[
                'dft_polarizability_time'] = timeit(
                    get_isotropic_polarizability, mf)
        except Exception as e:
            logger.error(
                f"{st.smiles} (id={st.unique_id}): Failed to calculate isotropic polarizability: {str(e)}"
            )

    # PySCF is not expected to change the order of atoms, but we update it just in case
    st.elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    st.atomic_numbers = [mol.atom_charge(i) for i in range(mol.natm)]

    st.metadata['dft_prop_total_time'] = round(time.perf_counter() - t_start,
                                               4)
    return st
