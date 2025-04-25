import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
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
assert version.parse(
    pyscf.__version__) > version.parse("2.5.0"), "Version must be > 2.5.0"

# try:
#     # check if GPU4PySCF is available
#     from gpu4pyscf.dft import rks
#     print("Using GPU-accelerated PySCF.")
# except ImportError:
#     print("GPU4PySCF not available, falling back to CPU.")
#     from pyscf.dft import rks

from pyscf.dft import rks
from pyscf.geomopt import geometric_solver

from .common import Structure
from .constants import EV_TO_HARTREE, COORDINATE_UNIT


@dataclass
class ASEOption:
    optimizer_name: str = 'BFGS'
    force_conv: float = 1e-5
    max_cycle: int = 1000


@dataclass
class PySCFOption:
    """
    Configuration options for PySCF geometry optimization.

    :param dft_functional: DFT functional to use.
    :param basis: Basis set to use.
    :param max_scf_cycle: Maximum number of SCF iterations allowed.
    :param scf_conv_tol: SCF convergence tolerance.
    :param grids_level: Level of grid refinement for numerical integration.
    :param use_gpu: Whether to use GPU for calculations. If None, auto-detect by:
                    if cupy is installed and a GPU is available, use gpu4pyscf;
                    otherwise, use normal pyscf CPU code.
    """
    basis: str = '6311g*'
    dft_functional: str = 'b3lypg'
    max_scf_cycle: int = 100  # default: 50 in pyscf
    scf_conv_tol: float = 1e-09  # default: 1e-09 in pyscf
    grids_level: int = 3  # default: 3 in pyscf
    use_gpu: Optional[bool] = None  # None means auto-detect


METHOD_AIMNet2 = 'aimnet2'
METHOD_DFT = 'dft'

OPTIMIZER_NAME_TO_CLASS = {'BFGS': BFGS, 'FIRE': FIRE}

PROPERTY_KEY_TEMPLATE = '{method}_{property_name}'


def optimize_by_aimnet2(st: Structure, options: ASEOption) -> Structure:
    """
    Optimize the geometry of a molecule using AIMNet2 calculator from ASE backend

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
        PROPERTY_KEY_TEMPLATE.format(method=METHOD_AIMNet2,
                                     property_name='energy'):
        float(ase_atoms.get_potential_energy()) * EV_TO_HARTREE,
        PROPERTY_KEY_TEMPLATE.format(method=METHOD_AIMNet2,
                                     property_name='charges'):
        ase_atoms.calc.results.get('charges', None),
        # TODO: is this necessary?
        PROPERTY_KEY_TEMPLATE.format(method=METHOD_AIMNet2,
                                     property_name='forces'):
        ase_atoms.get_forces().tolist(),
    }

    if st.atomic_numbers is None:
        st.atomic_numbers = ase_atoms.get_atomic_numbers().tolist()

    return st


def _detect_gpu() -> bool:
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


def optimize_by_pyscf(st: Structure, options: PySCFOption) -> Structure:
    """
    Optimize the geometry of a molecule using PySCF backend.

    :param st: Structure object containing the molecule information.
    :param options: PySCFOption object containing optimization parameters.
    :return: Structure object with optimized geometry and properties.
    """
    # Determine if we should use GPU
    use_gpu = options.use_gpu
    if use_gpu is None:
        use_gpu = _detect_gpu()

    # Setup molecule
    atom_str = '\n'.join(f'{el} {x} {y} {z}'
                         for el, (x, y, z) in zip(st.elements, st.xyz))
    mol = pyscf.gto.M(
        atom=atom_str,
        basis=options.basis,
        charge=st.charge,
        spin=st.multiplicity - 1,
        # Note mol.spin = 2S = Nalpha - Nbeta (not multiplicity=2S+1)
        unit=COORDINATE_UNIT)

    # Setup Kohn-Sham DFT object
    mf = rks.RKS(mol, xc=options.dft_functional).density_fit()
    mf.max_cycle = options.max_scf_cycle
    mf.conv_tol = options.scf_conv_tol
    mf.grids.level = options.grids_level

    gradients = []

    def callback(envs):
        gradients.append(envs['gradients'])

    # Optimize geometry with geomeTRIC library
    opt_start = time.perf_counter()
    is_converged, mol_optimized = geometric_solver.kernel(
        mf,
        callback=callback,
        maxsteps=500,  # default:100
    )
    if st.metadata is None:
        st.metadata = {}
    st.metadata['dft_opt_time'] = time.perf_counter() - opt_start

    if not is_converged:
        raise RuntimeError(
            f"Geometry optimization did not converge for molecule {st.unique_id} with SMILES {st.smiles}"
        )

    st.xyz = mol_optimized.atom_coords(unit=COORDINATE_UNIT)
    if st.property is None:
        st.property = {}
    st.property = {
        PROPERTY_KEY_TEMPLATE.format(method=METHOD_DFT, property_name='energy'):
        float(mf.e_tot),
        PROPERTY_KEY_TEMPLATE.format(method=METHOD_DFT, property_name='forces'):
        gradients[-1].tolist(),
    }

    return st
