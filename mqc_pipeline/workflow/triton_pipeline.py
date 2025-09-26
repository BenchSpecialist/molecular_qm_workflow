import os
import time
import yaml
import pickle
import multiprocessing
from enum import Enum
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

import numpy as np
from rdkit import Chem
from openbabel import pybel

from ..common import Structure
from ..smiles_util import smiles_to_structure_batch
from ..batch_optimize import FAILED_INPUTS_FILE, optimize_sts_by_triton
from .._aimnet import property_inference
from ..property import combustion_heat
from ..property.volume import get_vdw_volume

from ..settings import PySCFOption, ESPGridsOption, METHOD_AIMNet2, METHOD_DFT, ValidationError
from ..structure_io import write_molecule_property, write_atom_property, write_metadata
from ..util import get_default_logger, get_optimal_workers

logger = get_default_logger()

BATCH_SIZE_PER_GPU = 1024
TOTAL_CPUS_PER_NODE = 128

SUPPORTED_PROP_METHODS = [METHOD_AIMNet2, METHOD_DFT]


class AdditionalProperty(Enum):
    COMBUSTION_HEAT = "combustion_heat"
    VDW_VOLUME = "vdw_volume"


SUPPORTED_ADDITIONAL_PROPS = [
    AdditionalProperty.COMBUSTION_HEAT.value,
    AdditionalProperty.VDW_VOLUME.value
]


class TritonPipelineSettings(BaseModel):
    input_file: str = Field(
        description=
        "Path to the input file (CSV or txt) containing SMILES strings.")

    active_triton_nodes_file: str = Field(
        default="ACTIVE_TRITON_NODES.txt",
        description="Path to the file containing active Triton server nodes.\n"
        "# Each line should contain one node name (e.g., 'fs-sn-064').\n"
        "# If not specified, the program will check available active nodes first."
    )

    output_molecule_property_file: str = Field(
        default="molecule_property.csv",
        description="Output CSV/Parquet file to save molecule-level properties."
    )

    output_atom_property_file: str = Field(
        default="atom_property.csv",
        description=
        "Output CSV/Parquet file to save atom-level properties (XYZ, forces).")

    output_metadata_file: str | None = Field(
        default="metadata.csv",
        description=
        "Output CSV file to dump metadata (e.g., SMILES, unique_id, num_atoms, timings).\n"
        "# If set to `null`, metadata will not be saved.")

    property_method: str | None = Field(
        default=METHOD_AIMNet2,
        description="Method to compute molecular properties.\n"
        f"# Supported methods: {', '.join(SUPPORTED_PROP_METHODS)}")

    inference_max_batch_size: int = Field(
        default=2048,
        description=
        "Maximum number of structures in batch for parallel property inference."
    )

    inference_mo_range: tuple[float, float] = Field(
        default=(-20.0, 20.0),
        description=
        "Range for molecular orbital energies in property inference.\n"
        "# Outliers will be clipped to this range. Set to null to disable clipping."
    )

    inference_esp_range: tuple[float, float] = Field(
        default=(-20.0, 20.0),
        description="Range for electrostatic potential in property inference.\n"
        "# Outliers will be clipped to this range. Set to null to disable clipping."
    )

    post_infer_additional_properties: set[str] = Field(
        default=SUPPORTED_ADDITIONAL_PROPS,
        description=
        "Additional properties to compute after Aimnet2 property inference.\n"
        f"# Supported properties: {', '.join(SUPPORTED_ADDITIONAL_PROPS)}")

    dft_basis: str = Field(default='6311g*',
                           description="Basis set for PySCF calculations")
    dft_functional: str = Field(
        default='b3lypg', description="DFT functional for PySCF calculations")

    dft_grids_level: int = Field(
        default=4,
        description="Level of grid refinement for numerical integration")

    dft_solvent: bool | tuple[str, float] = Field(
        default=False,
        description="Whether to use solvent effects in PySCF calculations.\n"
        f"# If True, uses default IEF-PCM with epsilon=20.0.\n"
        "# If tuple[str, float], the first element is the solvent method (e.g., 'IEF-PCM')\n"
        "# and the second element is the solvent dielectric constant (e.g., 20.0)."
    )
    dft_dispersion: str | None = Field(
        default=None,
        description="Dispersion correction method for PySCF calculations.\n"
        "# Example: 'd3bj' (Grimme's D3 dispersion with Becke-Johnson damping), 'd3zero', 'd4'."
    )

    @field_validator('property_method')
    def validate_property_method(cls, method: str | None) -> str | None:
        if method is None:
            return None
        if method.upper() not in SUPPORTED_PROP_METHODS:
            raise ValidationError(
                f"Unsupported property calculation method: {method}. "
                f"Supported methods are: {', '.join(SUPPORTED_PROP_METHODS)}")
        return method

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TritonPipelineSettings":
        """
        Create a TritonPipelineSettings instance from a YAML file.
        """
        with open(yaml_path, "r") as fhandle:
            config_dict = yaml.safe_load(fhandle)
        # Validate and parse using the Pydantic model
        return cls(**config_dict)

    @classmethod
    def write_default_config_to_yaml(cls, yaml_path: str) -> None:
        """
        Write the default config YAML file with descriptions as comments.
        """
        schema = cls.model_json_schema()
        yaml_lines = []
        for key, val_dict in schema["properties"].items():
            description = val_dict.get("description", "")
            default_value = val_dict.get("default", " ")
            default_value = "null" if default_value is None else default_value
            if description:
                yaml_lines.append(f"# {description}")
            yaml_lines.append(f"{key}: {default_value}")
            yaml_lines.append("")  # Add a blank line for readability

        with open(yaml_path, "w") as file:
            file.write("\n".join(yaml_lines))

    def to_pyscf_options(self) -> PySCFOption:
        """
        Convert configuration settings to PySCFOption.

        :return: A PySCFOption object with settings from this configuration.
        """
        # Determine solvent settings based on dft_solvent value
        solvent_method, solvent_eps = (
            ('IEF-PCM', 20.0) if self.dft_solvent is True else
            self.dft_solvent if isinstance(self.dft_solvent, tuple) else
            (None, None))
        return PySCFOption(basis=self.dft_basis,
                           dft_functional=self.dft_functional,
                           grids_level=self.dft_grids_level,
                           dispersion=self.dft_dispersion,
                           solvent_method=solvent_method,
                           solvent_eps=solvent_eps)


def _log_failed_inputs(error_msg: str, error_file=FAILED_INPUTS_FILE) -> None:
    with open(error_file, 'a') as fp:
        fp.write(f'{error_msg}\n')


def _validate_single_smiles(smiles: str) -> tuple[str, bool, str | None]:
    """
    Validate a single SMILES string.

    :param smiles: SMILES string to validate
    :return: Tuple of (smiles, is_valid, error_message)
    """
    if (mol := Chem.MolFromSmiles(smiles)) is None:
        return smiles, False, "Invalid SMILES string"

    if (charge := sum(atom.GetFormalCharge() for atom in mol.GetAtoms())) != 0:
        return smiles, False, f"charge = {charge}, Triton server only supports neutral molecules"

    if (spin := sum(atom.GetNumRadicalElectrons()
                    for atom in mol.GetAtoms()) % 2) != 0:
        return smiles, False, f"num_unpaired_electron = {spin}, Triton server only supports closed-shell molecules"

    return smiles, True, None


def validate_smiles(smiles_list: list[str],
                    num_workers: int = TOTAL_CPUS_PER_NODE) -> list[str]:
    """
    Validate a list of SMILES strings in parallel and return only valid ones.

    :param smiles_list: List of SMILES strings to validate
    :param num_workers: Number of parallel workers to use

    :return: List of valid SMILES strings
    :raises ValueError: If no valid SMILES strings are found
    """
    t_start = time.perf_counter()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # map is synchronous, and blocks until all tasks are complete and returns all results
        results = pool.map(_validate_single_smiles, smiles_list)

    valid_smiles = [smiles for smiles, is_valid, _ in results if is_valid]
    [
        _log_failed_inputs(f"{smiles}: {error_msg}")
        for smiles, is_valid, error_msg in results if not is_valid
    ]

    if not valid_smiles:
        raise ValueError("No valid SMILES found.")

    logger.debug(
        f"validate_smiles ({num_workers} workers): {time.perf_counter() - t_start:.2f} seconds, "
        f"{len(valid_smiles)}/{len(smiles_list)} valid SMILES strings")
    return valid_smiles


def calc_combustion_heat_batch(sts: list,
                               energy_key: str = 'triton_energy_ev',
                               num_workers: int = TOTAL_CPUS_PER_NODE):
    """
    Calculate combustion heat for a batch of structures in parallel and add the
    combustion heat value to each structure's property dictionary.
    """
    t_start = time.perf_counter()

    element_combustion_data = combustion_heat.get_element_data_map(
        energy_key=energy_key)

    combustion_heat_func = partial(
        combustion_heat.calc_combustion_heat,
        element_combustion_data=element_combustion_data,
        dft_level=energy_key)

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(combustion_heat_func,
                               [(st, st.property.get(energy_key, 0.0))
                                for st in sts])

    # Add combustion heat values to each structure
    for st, (combustion_heat_val, _) in zip(sts, results):
        st.property['combustion_heat_ev'] = combustion_heat_val

    logger.info(
        f"calc_combustion_heat_batch ({num_workers} workers): {time.perf_counter() - t_start:.2f} seconds, "
        f"{len(results)} structures")


def add_vdw_volume_batch(sts: list, num_workers: int = TOTAL_CPUS_PER_NODE):
    """
    Calculate van der Waals volume for a batch of structures in parallel
    and add the vdw_volume value to each structure's property dictionary.
    Note that the vectorized CPU implementation of get_vdw_volume is used here.
    """
    t_start = time.perf_counter()

    vdw_volume_func = partial(get_vdw_volume, use_gpu=False)

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Output structures from Triton all have atomic_numbers attribute.
        results = pool.starmap(vdw_volume_func,
                               [(st.xyz, st.atomic_numbers) for st in sts])

    # Add vdw_volume values to each structure
    for st, vdw_volume in zip(sts, results):
        st.property['vdw_volume_angstrom3'] = vdw_volume

    logger.info(
        f"add_vdw_volume_batch ({num_workers} workers): {time.perf_counter() - t_start:.2f} seconds, "
        f"{len(results)} structures")


@dataclass(slots=True)
class SanitizeXYZOutput:
    is_valid: bool
    out_st: Structure | None = None
    error_message: str = ''


def sanitize_relaxed_xyz(st) -> SanitizeXYZOutput:
    obmol = pybel.readstring("xyz", st.to_xyz_block())
    ob_smiles = obmol.write("can").strip().split('\t')[0]

    rdkit_mol = Chem.MolFromSmiles(ob_smiles)
    if rdkit_mol is None:
        return SanitizeXYZOutput(
            is_valid=False,
            out_st=None,
            error_message=f"Relaxed XYZs-converted SMILES '{ob_smiles}' "
            "failed RDKit sanitization (Chem.MolFromSmiles returns None).")

    # Check if all bond lengths in the relaxed XYZs are within a reasonable range
    bond_distances = [
        np.linalg.norm(
            np.array(obmol.atoms[bond.GetBeginAtomIdx() - 1].coords) -
            np.array(obmol.atoms[bond.GetEndAtomIdx() - 1].coords))
        for bond in pybel.ob.OBMolBondIter(obmol.OBMol)
    ]
    unphysical_bond_lengths = [
        float(distance) for distance in bond_distances
        if distance < 0.5 or distance > 3.5
    ]
    if len(unphysical_bond_lengths) > 0:
        return SanitizeXYZOutput(
            is_valid=False,
            out_st=None,
            error_message=
            f"Unphysical bond distances in the OpenBabel geometry - {unphysical_bond_lengths}"
        )

    canonical_smiles = Chem.MolToSmiles(rdkit_mol,
                                        isomericSmiles=False,
                                        canonical=True)
    # Check if the molecule has multiple fragments
    if canonical_smiles.count(".") > 0:
        return SanitizeXYZOutput(
            is_valid=False,
            out_st=None,
            error_message=f"Disconnected fragments (. in SMILES).")

    st.property['canonical_smiles'] = canonical_smiles

    # Check if the molecule contains radical atoms
    if radical_electrons := {
        (atom.GetIdx(), atom.GetSymbol()): atom.GetNumRadicalElectrons()
            for atom in rdkit_mol.GetAtoms()
            if atom.GetNumRadicalElectrons() > 0
    }:
        st.property['num_radical_electrons'] = sum(radical_electrons.values())
        return SanitizeXYZOutput(
            is_valid=False,
            out_st=st,  # still save the structure but add error message
            error_message=f'Radical electrons found - {radical_electrons}')

    return SanitizeXYZOutput(is_valid=True, out_st=st)


def sanitize_xyz_parallel(sts: list,
                          num_workers: int = TOTAL_CPUS_PER_NODE) -> list:
    """
    Sanitize coordinates and generate canonical SMILES strings from a batch of structures
    in parallel; relaxed structures that failed to pass sanitization will be filtered out.
    """
    t_start = time.perf_counter()

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(sanitize_relaxed_xyz, sts)

    # Get output structures
    out_sts = [result.out_st for result in results if result.is_valid]

    # Collect structures with radical electrons and dump them to a file
    radical_sts = [
        result.out_st for result in results
        if 'Radical electrons' in result.error_message and not result.is_valid
    ]
    radical_st_dir = Path.cwd().parent / 'radical_sts'
    radical_st_dir.mkdir(parents=True, exist_ok=True)
    radical_st_outfile = radical_st_dir / Path(
        f'{Path.cwd().name}_{len(radical_sts)}_radical_sts.pkl')
    with open(radical_st_outfile, 'wb') as f:
        pickle.dump(radical_sts, f)
    logger.info(
        f"sanitize_xyz_parallel: {len(radical_sts)} molecules with radical electrons, saved to '{radical_st_outfile}'."
    )

    # Collect and log all error messages
    error_messages = [
        f"{st.smiles}: {result.error_message}"
        for st, result in zip(sts, results) if not result.is_valid
    ]
    _log_failed_inputs('\n'.join(error_messages))

    logger.info(
        f"sanitize_xyz_parallel ({num_workers} workers): {time.perf_counter() - t_start:.2f} seconds, "
        f"{len(out_sts)}/{len(sts)} outputs")

    return out_sts


def run_pipeline(inputs: list[str | Structure],
                 settings: TritonPipelineSettings):
    t_start = time.perf_counter()
    if len(inputs) == 0:
        logger.error("No inputs provided.")
        return

    # Get allocated CPUs from SLURM env, as multiprocessing.cpu_count() always
    # returns the total number of CPUs on a compute node, not the allocated ones.
    num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', TOTAL_CPUS_PER_NODE))
    # Auto-scale number of workers based on workload and CPU count
    max_workers = get_optimal_workers(total_inputs=len(inputs),
                                      min_items_per_worker=1000,
                                      cpu_count=num_cpus)

    init_sts_pkl = Path(f'init_sts.pkl')
    if isinstance(inputs[0], Structure):
        init_sts = inputs
    elif isinstance(inputs[0], str):
        # Validate SMILES strings in parallel
        smiles_list = validate_smiles(inputs, num_workers=max_workers)

        # Convert SMILES strings to 3D structures in parallel
        init_sts = smiles_to_structure_batch(smiles_list, num_workers=num_cpus)

        # Dump initial structures to a pickle file, as this is the most expensive step
        # saving initial structures helps reduce the time cost of re-running the pipeline
        with open(init_sts_pkl, 'wb') as f:
            pickle.dump(init_sts, f)

    # Optimize structures using Triton inference server
    opt_sts = optimize_sts_by_triton(init_sts, batch_size=BATCH_SIZE_PER_GPU)

    # Generate canonical SMILES strings from optimized coordinates
    # Relaxed XYZs that fail the sanitization are discarded.
    valid_opt_sts = sanitize_xyz_parallel(opt_sts, num_workers=num_cpus)

    # Get properties for optimized structures (ML inference or DFT)
    if settings.property_method == METHOD_AIMNet2:
        out_sts = property_inference.run_parallel(
            valid_opt_sts,
            mo_range=settings.inference_mo_range,
            esp_range=settings.inference_esp_range,
            max_batch_size=settings.inference_max_batch_size,
            num_cpu_workers=num_cpus)

        # Add combustion enthalpy (using Triton energy)
        if AdditionalProperty.COMBUSTION_HEAT.value in settings.post_infer_additional_properties:
            calc_combustion_heat_batch(out_sts,
                                       energy_key='triton_energy_ev',
                                       num_workers=num_cpus)

        # Add vdw_volume (using optimized structures)
        if AdditionalProperty.VDW_VOLUME.value in settings.post_infer_additional_properties:
            add_vdw_volume_batch(out_sts, num_workers=num_cpus)

    elif settings.property_method == METHOD_DFT:
        from ..property import get_properties_main
        pyscf_options = settings.to_pyscf_options()
        out_sts = [
            get_properties_main(
                st,
                pyscf_options=pyscf_options,
                esp_options=ESPGridsOption(),
                return_esp_range=True,
                return_combustion_heat=True,
            ) for st in valid_opt_sts
        ]
    else:
        # Skip property calculation if method is None
        out_sts = valid_opt_sts

    time_cost = time.perf_counter() - t_start
    avg_cost_time = time_cost / len(out_sts)
    logger.info(
        f"Total time: {time_cost:.2f} seconds to output {len(out_sts)} structures ({avg_cost_time:.2f} seconds per structure)"
    )
    logger.info(f'Success percentage: {len(out_sts) / len(inputs) * 100:.2f}%')

    mol_prop_outfile = Path(settings.output_molecule_property_file).resolve()
    atom_prop_outfile = Path(settings.output_atom_property_file).resolve()
    write_molecule_property(out_sts, filename=str(mol_prop_outfile))
    write_atom_property(out_sts, filename=str(atom_prop_outfile))

    logger.debug(
        f"Wrote molecule properties to {mol_prop_outfile}\n"
        f"Wrote atom properties (XYZ, forces) to {atom_prop_outfile}.")
    if settings.output_metadata_file:
        metadata_outfile = Path(settings.output_metadata_file).resolve()
        write_metadata(out_sts, filename=str(metadata_outfile))
        logger.debug(f"Wrote metadata to {metadata_outfile}")

    # Remove init_sts_pkl when the pipeline completes successfully
    if init_sts_pkl.exists():
        init_sts_pkl.unlink()
        logger.debug(f"Removed initial structures pickle file: {init_sts_pkl}")
    return out_sts
