import os
import time
import yaml
import multiprocessing
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from rdkit import Chem

from ..batch_optimize import FAILED_INPUTS_FILE, optimize_sts_by_triton
from .._aimnet import property_inference
from ..smiles_util import smiles_to_structure_batch
from ..property import get_properties_main
from ..settings import PySCFOption, ESPGridsOption, METHOD_AIMNet2, METHOD_DFT, ValidationError
from ..structure_io import write_molecule_property, write_atom_property
from ..util import get_default_logger, get_optimal_workers

logger = get_default_logger()

BATCH_SIZE_PER_GPU = 1024
TOTAL_CPUS_PER_NODE = 128

SUPPORTED_PROP_METHODS = [METHOD_AIMNet2, METHOD_DFT]


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

    property_method: str | None = Field(
        default=METHOD_AIMNet2,
        description="Method to compute molecular properties.\n"
        f"# Supported methods: {', '.join(SUPPORTED_PROP_METHODS)}")

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
        f"# If True, uses default IEF-PCM with epsilon=18.5.\n"
        "# If tuple[str, float], the first element is the solvent method (e.g., 'IEF-PCM')\n"
        "# and the second element is the solvent dielectric constant (e.g., 18.5)."
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
            ('IEF-PCM', 18.5) if self.dft_solvent is True else
            self.dft_solvent if isinstance(self.dft_solvent, tuple) else
            (None, None))
        return PySCFOption(basis=self.dft_basis,
                           dft_functional=self.dft_functional,
                           grids_level=self.dft_grids_level,
                           dispersion=self.dft_dispersion,
                           solvent_method=solvent_method,
                           solvent_eps=solvent_eps)


def _log_failed_inputs(error_msg: str, error_file=FAILED_INPUTS_FILE) -> None:
    logger.error(error_msg)
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
        f"valid SMILES strings {len(valid_smiles)}/{len(smiles_list)}")
    return valid_smiles


def run_pipeline(smiles_list: list[str], settings: TritonPipelineSettings):
    t_start = time.perf_counter()

    # Get allocated CPUs from SLURM env, as multiprocessing.cpu_count() always
    # returns the total number of CPUs on a compute node, not the allocated ones.
    num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', TOTAL_CPUS_PER_NODE))
    # Auto-scale number of workers based on workload and CPU count
    max_workers = get_optimal_workers(total_inputs=len(smiles_list),
                                      min_items_per_worker=1000,
                                      cpu_count=num_cpus)

    # Validate SMILES strings in parallel
    smiles_list = validate_smiles(smiles_list, num_workers=max_workers)

    # Convert SMILES strings to 3D structures in parallel
    init_sts = smiles_to_structure_batch(smiles_list, num_workers=num_cpus)

    # Optimize structures using Triton inference server
    opt_sts = optimize_sts_by_triton(init_sts, batch_size=BATCH_SIZE_PER_GPU)

    # Get properties for optimized structures (ML inference or DFT)
    if settings.property_method == METHOD_AIMNet2:
        out_sts = property_inference.run(
            opt_sts,
            mo_range=property_inference.MO_CLIP_RANGE,
            esp_range=property_inference.ESP_CLIP_RANGE)

    elif settings.property_method == METHOD_DFT:
        pyscf_options = settings.to_pyscf_options()
        out_sts = [
            get_properties_main(
                st,
                pyscf_options=pyscf_options,
                esp_options=ESPGridsOption(),
                return_esp_range=True,
                return_combustion_heat=True,
            ) for st in opt_sts
        ]

    logger.info(
        f"Total time: {len(out_sts)} structures in {time.perf_counter() - t_start:.4f} seconds"
    )

    mol_prop_outfile = Path('molecule_property.csv').resolve()
    atom_prop_outfile = Path('atom_property.csv').resolve()
    write_molecule_property(out_sts,
                            filename=str(mol_prop_outfile),
                            additional_mol_keys=[])
    write_atom_property(out_sts, filename=str(atom_prop_outfile))

    logger.info(f"Wrote molecule properties to {mol_prop_outfile}\n"
                f"Wrote atom properties (XYZ, forces) to {atom_prop_outfile}.")
    return out_sts
