import time
from pathlib import Path
from functools import lru_cache
from ase.optimize import BFGS, FIRE
from pyscf.geomopt import geometric_solver

try:
    from .aimnet2calc.aimnet2ase import AIMNet2ASE
except Exception as e:
    AIMNet2ASE = None

from .common import Structure, setup_mean_field_obj, COORDINATE_UNIT
from .property.keys import DFT_ENERGY_KEY, DFT_FORCES_KEY
from .settings import ASEOption, PySCFOption, METHOD_AIMNet2
from .util import get_default_logger

logger = get_default_logger()

OPTIMIZER_NAME_TO_CLASS = {'BFGS': BFGS, 'FIRE': FIRE}

_DEFAULT_AIMNET2_MODEL_PATH = Path(
    __file__).parent / "models" / "aimnet2_wb97m_0.jpt"


@lru_cache(maxsize=1)
def _get_aimnet2calc(model_path: Path | str):
    """
    Load the local AIMNet2 model checkpoint once and cache it for subsequent calls.
    """
    try:
        from .aimnet2calc.calculator import AIMNet2Calculator
    except Exception as e:
        err_msg = f"Cannot import aimnet2calc.AIMNet2Calculator: {str(e)}"
        logger.error(err_msg)
        raise ImportError(err_msg)

    logger.info(f'Loading AIMNet2 model from {model_path}')
    return AIMNet2Calculator(str(model_path))


# def _runs_with_slurm() -> bool:
#     """
#     Check if the current Python process is running under SLURM orchestration.
#     """
#     slurm_vars = ['SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_LOCALID']
#     return any(var in os.environ for var in slurm_vars)


def optimize_by_aimnet2(st: Structure,
                        options: ASEOption,
                        model_path: Path | None = None) -> Structure:
    """
    Optimize the geometry of a molecule using AIMNet2 calculator from ASE backend.
    Note that the function modifies the input Structure object in-place.

    :param st: Structure object containing the molecule information.
    :param options: ASEOption object containing optimization parameters.
    """
    model_path = model_path or _DEFAULT_AIMNET2_MODEL_PATH
    model_path = str(model_path)

    t_start = time.perf_counter()
    ase_atoms = st.to_ase_atoms()

    if AIMNet2ASE is None:
        raise ImportError(
            "AIMNet2ASE is not available. Check if aimnet2calc package is installed."
        )

    ase_atoms.calc = AIMNet2ASE(
        base_calc=_get_aimnet2calc(
            model_path),  # base_calc='AIMNET2' fetch online model
        charge=st.charge,
        mult=st.multiplicity)

    # Set algorithm/optimizer used for geometry optimization
    if options.optimizer_name not in OPTIMIZER_NAME_TO_CLASS:
        raise ValueError(
            f"Optimizer {options.optimizer_name} is not supported. "
            f"Supported optimizers are: {', '.join(OPTIMIZER_NAME_TO_CLASS.keys())}"
        )
    optimizer = OPTIMIZER_NAME_TO_CLASS[options.optimizer_name](ase_atoms)

    # Run the optimization
    optimizer.run(fmax=options.force_conv, steps=options.max_cycle)

    # Check if the optimization was successful
    if not optimizer.converged():
        raise RuntimeError(
            f"Optimization did not converge for molecule {st.unique_id} with SMILES {st.smiles}"
        )

    # Save optimization time to metadata
    st.metadata['aimnet2_opt_time'] = round(time.perf_counter() - t_start, 4)

    # Update the input structure with the optimized geometry
    st.xyz = ase_atoms.get_positions()

    # Save the AIMNet2 energy and forces of the optimized geometry
    st.property[f"{METHOD_AIMNet2}_energy_ev"] = float(
        ase_atoms.get_potential_energy())
    st.save_gradients(ase_atoms.get_forces(),
                      prop_key=f"{METHOD_AIMNet2}_forces")

    charges = ase_atoms.calc.results.get('charges')
    if charges is not None:
        st.atom_property[f"{METHOD_AIMNet2}_charges"] = charges

    st.atomic_numbers = ase_atoms.get_atomic_numbers().tolist()

    return st


def optimize_by_pyscf(st: Structure,
                      options: PySCFOption,
                      save_metadata: bool = False) -> Structure:
    """
    Optimize the geometry of a molecule using PySCF backend.

    :param st: Structure object containing the molecule information
    :param options: PySCFOption object with relevant parameters to set up PySCF
    :param save_metadata: Whether to save optimization trajectory metadata
    :return: Structure object with optimized geometry; the energy and forces of
             the equilibrium geometry are saved in the `property` attribute
    """
    # Setup molecule
    mol = st.to_pyscf_mole(basis=options.basis)

    # Setup Kohn-Sham DFT object
    mf = setup_mean_field_obj(mol, options)

    # Save energies and gradients along the optimization trajectory
    energies, gradients = [], []

    def callback(envs):
        energies.append(envs['energy'])
        gradients.append(envs['gradients'])

    # Optimize geometry with geomeTRIC library
    t_start = time.perf_counter()
    is_converged, mol_optimized = geometric_solver.kernel(
        mf,
        callback=callback,
        maxsteps=500,  # default:100
    )
    # Save optimization time to metadata
    st.metadata['dft_opt_time'] = round(time.perf_counter() - t_start, 4)
    if save_metadata:
        st.metadata['dft_opt_energies'] = energies
        st.metadata['dft_opt_gradients'] = gradients

    if not is_converged:
        raise RuntimeError(
            f"Geometry optimization did not converge for molecule {st.unique_id} with SMILES {st.smiles}"
        )

    # Update the input structure with the optimized coordinates
    st.xyz = mol_optimized.atom_coords(unit=COORDINATE_UNIT)

    # Save the energy and forces of the optimized geometry
    st.property[DFT_ENERGY_KEY] = float(energies[-1])
    st.save_gradients(gradients[-1], prop_key=DFT_FORCES_KEY)

    # PySCF is not expected to change the order of atoms, but we update it just in case
    st.elements = [
        mol_optimized.atom_symbol(i) for i in range(mol_optimized.natm)
    ]
    st.atomic_numbers = mol_optimized.atom_charges().tolist()

    return st
