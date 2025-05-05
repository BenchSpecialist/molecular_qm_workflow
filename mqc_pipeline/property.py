try:
    # check if GPU4PySCF is available
    from gpu4pyscf.dft import rks
    from gpu4pyscf.qmmm import chelpg
    from gpu4pyscf.gto.int3c1e import int1e_grids
    print("Using GPU-accelerated PySCF.\n")
except ImportError:
    print("GPU4PySCF not available, falling back to normal CPU PySCF.\n")
    from pyscf.dft import rks

from .common import Structure, COORDINATE_UNIT
from .constants import HARTREE_TO_EV, ANGSTROM_TO_BOHR
from .settings import PySCFOption


def get_properties_by_pyscf(st: Structure,
                            options: PySCFOption,
                            return_gradient: bool = False) -> Structure:
    """
    Compute properties for the given structure using PySCF backend.

    :param st: Structure object containing the molecule information.
    :param options: PySCFOption object with relevant parameters to set up PySCF.
    :param return_gradient: Whether to run gradient calculations. This needs to be
                            true if the input structure is not optimized by PySCF,
                            as the DFT gradients are not available from the geometry
                            optimization.

    :return: Structure object with populated `property` attribute.
    """
    mol = st.to_pyscf_mole()
    mol.basis = options.basis

    mf = rks.RKS(mol, xc=options.dft_functional).density_fit()
    mf.max_cycle = options.max_scf_cycle
    mf.conv_tol = options.scf_conv_tol
    mf.grids.level = options.grids_level

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
    st.property['dipole_moment_debye'] = mf.dip_moment(unit='Debye', dm=rdm1)
    # grids = get_esp_grid(mol,
    #                      rcut=option.solvent_accessible_region,
    #                      space=self.grid_spacing,
    #                      probe=self.probe)
    # esp = int1e_grids(mol, grids, dm=rdm1)
    # st.property['esp_min_eV'] = min(esp) * HARTREE_TO_EV
    # st.property['esp_max_eV'] = min(esp) * HARTREE_TO_EV

    if return_gradient:
        st.property["forces"] = mf.Gradients().kernel()

    # Evaluate CHELPG charges and transfers data from GPU (cupy) to CPU (numpy)
    chelpg_charges = chelpg.eval_chelpg_layer_gpu(mf).get()
    st.property['chelpg_charges'] = [
        # type: (atom_symbol: string, charge: float)
        (mol.atom_symbol(i), float(chelpg_charges[i])) for i in range(mol.natm)
    ]
