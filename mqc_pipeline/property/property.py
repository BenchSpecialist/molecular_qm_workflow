from gpu4pyscf.dft import rks, uks
from gpu4pyscf.qmmm import chelpg

from ..common import Structure
from ..constants import HARTREE_TO_EV
from ..settings import PySCFOption, ESPGridsOption

from .esp import generate_esp_grids, get_esp_range


def get_properties_neutral(st: Structure,
                           pyscf_options: PySCFOption,
                           esp_options: ESPGridsOption,
                           return_gradient: bool = False) -> Structure:
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
    assert st.charge == 0, "This function is for neutral molecules only."

    mol = st.to_pyscf_mole()
    mol.basis = pyscf_options.basis
    mol.build()

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

    # Get the total SCF energy
    st.property['energy_hartree'] = float(mf.e_tot)

    # Get HOMO and LUMO energies (neutral species only) in eV.
    st.property['homo_eV'] = float(
        mf.mo_energy[mf.mo_occ > 0][-1]) * HARTREE_TO_EV
    st.property['lumo_eV'] = float(
        mf.mo_energy[mf.mo_occ == 0][0]) * HARTREE_TO_EV

    # Compute one-body reduced density matrix for dipole, electrostatic potential calculations
    rdm1 = mf.make_rdm1()
    st.property['dipole_x_debye'], st.property['dipole_y_debye'], st.property[
        'dipole_z_debye'] = mf.dip_moment(unit='Debye', dm=rdm1).tolist()

    # Generate grids for ESP calculations
    grids = generate_esp_grids(mol,
                               rcut=esp_options.solvent_accessible_region,
                               space=esp_options.grid_spacing,
                               solvent_probe=esp_options.probe_depth)
    st.property['esp_min_eV'], st.property['esp_max_eV'] = get_esp_range(
        mol, grids, one_rdm=rdm1)

    if return_gradient:
        gradients_arr = mf.Gradients().kernel()
        st.save_gradients(gradients_arr)

    # Evaluate CHELPG charges and transfers data from GPU (cupy) to CPU (numpy)
    chelpg_charges = chelpg.eval_chelpg_layer_gpu(mf).get()
    st.save_charges(chelpg_charges, prop_key='chelpg_charge')

    return st
