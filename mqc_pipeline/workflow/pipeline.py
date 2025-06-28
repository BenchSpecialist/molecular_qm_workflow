import os
import time
from pathlib import Path
from typing import Callable, Optional
from functools import partial

from ..settings import PipelineSettings, METHOD_AIMNet2, METHOD_DFT
from ..common import Structure
from ..smiles_util import smiles_to_structure
from .. import optimize
from ..property import get_properties_main
from ..structure_io import write_molecule_property, write_atom_property
from ..util import get_default_logger, setup_logger

from .io import read_smiles, read_xyz_dir

logger = get_default_logger()
_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES") or 0

MOL_PROP_OUTFILE = "molecule_property.{ext}"
ATOM_PROP_OUTFILE = "atom_property.{ext}"

FAILED_INPUTS = Path("FAILED_INPUTS.txt")


def _log_failed_inputs(error_msg: str) -> None:
    logger.error(error_msg)
    with FAILED_INPUTS.open('a') as fp:
        fp.write(f'{error_msg}\n')


def run_one_molecule(smiles_or_st: str | Structure,
                     prop_func: Callable,
                     smiles_to_3d_method: Optional[str] = None,
                     opt_func: Optional[Callable] = None) -> Structure | None:
    t_start = time.perf_counter()
    if isinstance(smiles_or_st, Structure):
        st = smiles_or_st
    elif isinstance(smiles_or_st, str):
        smiles = smiles_or_st
        # 3D structure generation from SMILES
        try:
            st = smiles_to_structure(smiles, method=smiles_to_3d_method)
        except Exception as e:
            err_msg = f"{smiles}: 3D structure generation failed - {str(e)}"
            _log_failed_inputs(err_msg)
            return
        logger.info(f"{smiles}: SMILES to 3D structure generation succeeded.")
    else:
        raise RuntimeError(
            "Input must be a SMILES string or a Structure object.")

    # Geometry optimization
    if opt_func is not None:
        try:
            st = opt_func(st)
        except Exception as e:
            err_msg = f"{st.unique_id} (smiles: {st.smiles}): Geometry optimization failed - {str(e)}"
            _log_failed_inputs(err_msg)
            return
        logger.info(
            f"{st.smiles} (id={st.unique_id}): Geometry optimization converged."
        )
    else:
        logger.info(
            f"{st.smiles} (id={st.unique_id}): Skipped geometry optimization.")

    # Property calculation
    try:
        st = prop_func(st)
    except Exception as e:
        err_msg = f"{st.unique_id} (smiles: {st.smiles}): Property calculation failed - {str(e)}"
        _log_failed_inputs(err_msg)
    logger.info(
        f"{st.smiles} (id={st.unique_id}): Property calculations done.")

    st.metadata['total_time'] = round(time.perf_counter() - t_start, 4)
    return st


def run_one_batch(inputs: list[str] | list[Structure],
                  settings: PipelineSettings):
    """
    Run pipeline for a batch of molecules; they are intended to run in serial
    on a single GPU.
    """
    progress_logger = setup_logger("progress_logger", log_file=f"PROGRESS.log")
    pyscf_options = settings.to_pyscf_options()

    # Get additional property calculation kwargs from settings
    additional_prop_kwargs = settings.get_property_kwargs()

    # Set backend used for geometry optimization
    if settings.geometry_opt_method is None:
        opt_func = None
    elif settings.geometry_opt_method.upper() == METHOD_AIMNet2:
        opt_func = partial(optimize.optimize_by_aimnet2,
                           options=settings.to_ase_options())
    elif settings.geometry_opt_method.upper() == METHOD_DFT:
        opt_func = partial(optimize.optimize_by_pyscf, options=pyscf_options)
        # DFT-optimized structures already save the gradient info.
        additional_prop_kwargs.update({'return_gradient': False})

    # Set up property calculator, get all properties
    prop_func = partial(get_properties_main,
                        pyscf_options=pyscf_options,
                        esp_options=settings.to_esp_grids_options(),
                        **additional_prop_kwargs)

    # Run the pipeline for each molecule sequentially
    out_sts = []
    total_count = len(inputs)

    for i, smiles_or_st in enumerate(inputs):
        # The function does error handling/logging internally
        # and returns None if it fails.
        st = run_one_molecule(
            smiles_or_st,
            smiles_to_3d_method=settings.smiles_to_3d_method.lower(),
            opt_func=opt_func,
            prop_func=prop_func)

        if st is None:
            progress_logger.info(f"{i + 1}/{total_count} FAILED")
        else:
            out_sts.append(st)
            # Log progress at specified intervals
            if (i + 1) % settings.progress_log_interval == 0 \
                or i + 1 == total_count:
                progress_logger.info(
                    f"{i + 1}/{total_count} DONE in {st.metadata['total_time']} seconds"
                )

    ext = settings.output_file_format.lower()
    if len(out_sts) == 0:
        logger.warning("No successful structures processed. Exiting.")
        return
    # Write molecule-level properties to a file
    mol_prop_out = MOL_PROP_OUTFILE.format(ext=ext)
    write_molecule_property(out_sts, mol_prop_out)
    logger.info(
        f"Molecule-level properties written to {os.getcwd()}/{mol_prop_out}")

    # Write atom-level properties to a separate file
    atom_prop_out = ATOM_PROP_OUTFILE.format(ext=ext)
    write_atom_property(out_sts, atom_prop_out)
    logger.info(
        f"Atom-level properties written to {os.getcwd()}/{atom_prop_out}")


def run_from_config_settings(settings: PipelineSettings) -> None:
    # Read input from config
    input_path = Path(settings.input_file_or_dir)
    if input_path.is_file():
        # type: list[str]
        inputs = read_smiles(input_path)
        logger.info(f"Read {len(inputs)} SMILES from {input_path}")
    else:
        # type: list[Structure]
        inputs = list(read_xyz_dir(input_path))

    # Run the pipeline
    run_one_batch(inputs, settings)
