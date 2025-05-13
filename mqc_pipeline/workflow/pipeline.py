import os
import logging
from pathlib import Path
from typing import Callable
from functools import partial

from ..settings import PipelineSettings, METHOD_AIMNet2, METHOD_DFT
from ..common import Structure
from ..smiles_util import smiles_to_3d_structures_by_rdkit
from .. import optimize
from ..property import get_properties_neutral
from ..structure_io import write_molecule_property, write_structure_atom_property

from .io import read_smiles, read_xyz_dir

_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES") or 0

MOL_PROP_BATCH_OUTFILE = "{GPU_ID}_{mol_id_of_first_mol}_molecule_property.csv"
ATOM_PROP_OUTFILE = "{mol_id}_atom_property.csv"

outfile_doc = """Output files are saved in the working directory with the following naming convention:
- {GPU_ID}_{mol_id}_molecule_property.csv:
    one file saving molecule-level properties of all molecules in the batch.
- {mol_id}_atom_property.csv:
    atom-level properties, each molecule has its own file.
"""


def run_one_molecule(smiles_or_st: str | Structure, opt_func: Callable,
                     prop_func: Callable) -> Structure:

    if isinstance(smiles_or_st, Structure):
        st = smiles_or_st
    elif isinstance(smiles_or_st, str):
        smiles = smiles_or_st
        # 3D structure generation from SMILES
        try:
            st = smiles_to_3d_structures_by_rdkit(smiles)
        except Exception as e:
            logging.error(f"Error generating 3D structure for {smiles}: {e}")
            return
    else:
        raise ValueError(
            "Input must be a SMILES string or a Structure object.")

    # Geometry optimization
    try:
        st = opt_func(st)
    except Exception as e:
        logging.error(
            f"Error optimizing geometry for {st.unique_id} (smiles: {st.smiles}): {e}"
        )
        return

    # Property calculation
    try:
        st = prop_func(st)
    except Exception as e:
        logging.error(
            f"Error calculating properties for {st.unique_id} (smiles: {st.smiles}): {e}"
        )

    return st


def run_one_batch(inputs: list[str] | list[Structure],
                  settings: PipelineSettings):
    """
    Run pipeline for a batch of molecules; they are intended to run in serial
    on a single GPU.
    """
    pyscf_options = settings.to_pyscf_options()
    esp_options = settings.to_esp_grids_options()

    # Set backend used for geometry optimization
    if settings.geometry_opt_method.upper() == METHOD_AIMNet2:
        opt_func = partial(optimize.optimize_by_aimnet2,
                           options=settings.to_ase_options)
    if settings.geometry_opt_method.upper() == METHOD_DFT:
        opt_func = partial(optimize.optimize_by_pyscf, options=pyscf_options)

    # Set up property calculator
    prop_func = partial(get_properties_neutral,
                        pyscf_options=pyscf_options,
                        esp_options=esp_options)

    # Run the pipeline for each molecule sequentially
    out_sts = []
    for smiles_or_st in inputs:
        # Put detailed error handling in the function for clarity
        st = run_one_molecule(smiles_or_st, opt_func, prop_func)
        if st:
            out_sts.append(st)

    # Write results to files
    mol_prop_outfile = MOL_PROP_BATCH_OUTFILE.format(
        GPU_ID=_GPU_ID, mol_id_of_first_mol=out_sts[0].unique_id)
    write_molecule_property(out_sts, mol_prop_outfile)
    for st in out_sts:
        write_structure_atom_property(
            st, ATOM_PROP_OUTFILE.format(mol_id=st.unique_id))


def run_from_config_settings(settings: PipelineSettings) -> None:
    logging.info(outfile_doc)

    # Read input from config
    input_path = Path(settings.input_file_or_dir)
    if input_path.is_file():
        # type: list[str]
        inputs = read_smiles(input_path)
        logging.info(f"Read {len(inputs)} SMILES from {input_path}")
    else:
        # type: list[Structure]
        inputs = list(read_xyz_dir(input_path))

    # Run the pipeline
    run_one_batch(inputs, settings)
