import time

from packaging import version

# Imports for ASE backend
from ase.optimize import BFGS, FIRE
try:
    from aimnet2calc import AIMNet2ASE
except ImportError:
    AIMNet2ASE = None
    print("AIMNet2 calculator is not available. "
          "Please install the required package.")

# Imports for pyscf backend
import pyscf
from pyscf.geomopt import geometric_solver
assert version.parse(
    pyscf.__version__) > version.parse("2.5.0"), "Version must be > 2.5.0"

try:
    # check if GPU4PySCF is available
    from gpu4pyscf.dft import rks, uks
    print("Using GPU-accelerated PySCF.\n")
except ImportError:
    print("GPU4PySCF not available, falling back to normal CPU PySCF.\n")
    from pyscf.dft import rks, uks

from .common import Structure, COORDINATE_UNIT
from .constants import EV_TO_HARTREE
from .settings import ASEOption, PySCFOption, METHOD_AIMNet2

OPTIMIZER_NAME_TO_CLASS = {'BFGS': BFGS, 'FIRE': FIRE}


def optimize_by_aimnet2(st: Structure, options: ASEOption) -> Structure:
    """
    Optimize the geometry of a molecule using AIMNet2 calculator from ASE backend.
    Note that the function modifies the input Structure object in-place.

    :param st: Structure object containing the molecule information.
    :param options: ASEOption object containing optimization parameters.
    """
    # Attach AIMNet2 calculator to the molecule
    ase_atoms = st.to_ase_atoms()
    ase_atoms.calc = AIMNet2ASE(base_calc=METHOD_AIMNet2,
                                charge=st.charge,
                                mult=st.multiplicity)

    # Set algorithm/optimizer used for geometry optimization
    if options.optimizer_name not in OPTIMIZER_NAME_TO_CLASS:
        raise ValueError(
            f"Optimizer {options.optimizer_name} is not supported. "
            f"Supported optimizers are: {list(OPTIMIZER_NAME_TO_CLASS.keys())}"
        )
    optimizer = OPTIMIZER_NAME_TO_CLASS[options.optimizer_name](ase_atoms)

    # Run the optimization
    optimizer.run(fmax=options.force_conv, steps=options.max_cycle)

    # Check if the optimization was successful
    if not optimizer.converged():
        raise RuntimeError(
            f"Optimization did not converge for molecule {st.unique_id} with SMILES {st.smiles}"
        )

    # Update the input structure with the optimized geometry
    st.xyz = ase_atoms.get_positions()
    st.property = {
        f"{METHOD_AIMNet2}_energy_hartree":
        float(ase_atoms.get_potential_energy()) * EV_TO_HARTREE,
        f"{METHOD_AIMNet2}_charges":
        ase_atoms.calc.results.get('charges', None),
        f"{METHOD_AIMNet2}_forces":
        ase_atoms.get_forces().tolist(),
    }

    if st.atomic_numbers is None:
        st.atomic_numbers = ase_atoms.get_atomic_numbers().tolist()

    return st


def _is_gpu4pyscf_compatible() -> bool:
    """
    Detect if cupy is installed (GPU4PySCF requires cupy) and if a CUDA-compatible
    GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    try:
        import cupy
        _ = cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, AttributeError):
        # cupy not installed or not configured correctly
        return False
    except Exception:
        # Any other error indicates no GPU access
        return False


def optimize_by_pyscf(st: Structure,
                      options: PySCFOption,
                      save_metadata: bool = False) -> Structure:
    """
    Optimize the geometry of a molecule using PySCF backend.

    :param st: Structure object containing the molecule information.
    :param options: PySCFOption object with relevant parameters to set up PySCF.

    :return: Structure object with optimized geometry; the energy and forces of
             the equilibrium geometry are saved in the `property` attribute.
    """
    # Setup molecule
    mol = st.to_pyscf_mole()
    mol.basis = options.basis
    # Refresh internal data of Mole object when modifying its attribute.
    mol.build()

    # Setup Kohn-Sham DFT object
    if st.multiplicity == 1:  # closed-shell
        mf = rks.RKS(mol, xc=options.dft_functional).density_fit()
    else:
        mf = uks.UKS(mol, xc=options.dft_functional).density_fit()

    mf.max_cycle = options.max_scf_cycle
    mf.conv_tol = options.scf_conv_tol
    mf.grids.level = options.grids_level

    # Save energies and gradients along the optimization trajectory
    energies, gradients = [], []

    def callback(envs):
        energies.append(envs['energy'])
        gradients.append(envs['gradients'])

    # Optimize geometry with geomeTRIC library
    opt_start = time.perf_counter()
    is_converged, mol_optimized = geometric_solver.kernel(
        mf,
        callback=callback,
        maxsteps=500,  # default:100
    )
    # Save optimization time to metadata
    if st.metadata is None:
        st.metadata = {}
    st.metadata['dft_opt_time'] = time.perf_counter() - opt_start
    if save_metadata:
        st.metadata['dft_opt_energies'] = energies
        st.metadata['dft_opt_gradients'] = gradients

    if not is_converged:
        raise RuntimeError(
            f"Geometry optimization did not converge for molecule {st.unique_id} with SMILES {st.smiles}"
        )

    # Update the input structure with the optimized coordinates
    st.xyz = mol_optimized.atom_coords(unit=COORDINATE_UNIT)
    if st.property is None:
        st.property = {}

    st.property = {
        "energy_hartree": float(energies[-1]),
        "forces": gradients[-1].tolist(),
    }

    return st
