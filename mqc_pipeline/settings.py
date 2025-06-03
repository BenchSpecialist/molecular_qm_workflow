import yaml
from enum import Enum
from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, model_validator

### Module constants ###
METHOD_AIMNet2 = 'AIMNET2'
METHOD_DFT = 'DFT'
SUPPORTED_GEOMETRY_OPT_METHODS = [METHOD_AIMNet2, METHOD_DFT]

BFGS_OPTIMIZER = "BFGS"
FIRE_OPTIMIZER = "FIRE"
SUPPORTED_ASE_OPTIMIZERS = (BFGS_OPTIMIZER, FIRE_OPTIMIZER)

_DEFAULT_BASIS = '6311g*'
_DEFAULT_FUNCTIONAL = 'b3lypg'
_DEFAULT_SCF_MAX_CYCLE = 100  # default: 50 in pyscf
_DEFAULT_SCF_CONV_TOL = 1e-09  # default: 1e-09 in pyscf
_DEFAULT_GRIDS_LEVEL = 3  # default: 3 in pyscf

SUPPORTED_COLUMNAR_FILE_FORMATS = ('csv', 'parquet')


class ValidationError(Exception):
    pass


class AdditionalProperty(Enum):
    """Supported additional properties."""
    COMBUSTION_HEAT = "combustion_heat"
    ESP_RANGE = "esp_range"
    CHELPG_CHARGES = "chelpg_charges"
    FORCES = "forces"
    FREQ = "freq"
    VDW_VOLUME = "vdw_volume"
    ISOTROPIC_POLARIZABILITY = "isotropic_polarizability"
    QUADRUPOLE = "quadrupole"

    @classmethod
    def _default_props_solvent(cls) -> set[str]:
        return {
            cls.COMBUSTION_HEAT.value, cls.CHELPG_CHARGES.value,
            cls.ESP_RANGE.value
        }

    @classmethod
    def to_kwargs_mapping(cls) -> dict[str, str]:
        """Get mapping from property names to function parameter names."""
        return {
            cls.COMBUSTION_HEAT.value: "return_combustion_heat",
            cls.ESP_RANGE.value: "return_esp_range",
            cls.CHELPG_CHARGES.value: "return_chelpg_chg",
            cls.FORCES.value: "return_gradient",
            cls.FREQ.value: "return_freq",
            cls.VDW_VOLUME.value: "return_volume",
            cls.ISOTROPIC_POLARIZABILITY.value: "return_polarizability",
            cls.QUADRUPOLE.value: "return_quadrupole"
        }


SUPPORTED_ADDITIONAL_PROPS = tuple(prop.value for prop in AdditionalProperty)


@dataclass(slots=True)
class ASEOption:
    """
    Configuration options for ASE backend.

    :param optimizer_name: Name of the optimizer to use.
    :param force_conv: Convergence threshold for forces.
    :param max_cycle: Maximum number of optimization cycles allowed.
    """
    optimizer_name: str = BFGS_OPTIMIZER
    force_conv: float = 1e-5
    max_cycle: int = 1000


@dataclass(slots=True)
class PySCFOption:
    """
    Configuration options for PySCF backend.

    :param dft_functional: DFT functional to use.
    :param basis: Basis set to use.
    :param max_scf_cycle: Maximum number of SCF iterations allowed.
    :param scf_conv_tol: SCF convergence tolerance.
    :param grids_level: Level of grid refinement for numerical integration used
                        in DFT to evaluate the exchange-correlation energy and
                        potential. This setting is DFT-relevant, and not used for
                        wavefunction-based methods like Hartree-Fock, MP2 or CCSD.
    """
    basis: str = _DEFAULT_BASIS
    dft_functional: str = _DEFAULT_FUNCTIONAL
    max_scf_cycle: int = _DEFAULT_SCF_MAX_CYCLE
    scf_conv_tol: float = _DEFAULT_SCF_CONV_TOL
    grids_level: int = _DEFAULT_GRIDS_LEVEL


@dataclass(slots=True)
class ESPGridsOption:
    """
    Configuration options for ESP grid generation.

    :param solvent_accessible_region: Solvent accessible region for ESP calculations in angstrom.
    :param grid_spacing: Grid spacing for ESP calculations in angstrom.
    :param probe_depth: Probe depth for ESP calculations in angstrom.
    """
    solvent_accessible_region: float = 3.0
    grid_spacing: float = 0.5
    probe_depth: float = 1.1


class PipelineSettings(BaseModel):
    """
    Frontend schema for the pipeline settings.

    Example usage:
    >>> user_dict = {"input_file_or_dir": "smiles.txt"}
    >>> config = PipelineSettings(**user_dict)
    >>> config_dict = config.model_dump() # Convert to a dictionary
    """
    # Input and job settings
    input_file_or_dir: str = Field(
        description=
        "Path to a text/csv file that contains a single column of smiles strings.\n"
        "# Alternatively, a directory containing multiple xyz files.")

    num_jobs: int = Field(
        default=1,
        ge=0,
        description="Number of SLURM batch jobs to launch.\n"
        "# If set to 0, the pipeline will run locally without SLURM orchestration."
    )

    job_name: str = Field(
        default="mqc_pipeline",
        description="Name of the SLURM job. Only relevant when num_jobs > 0.")

    # Calculation parameters
    geometry_opt_method: str | None = Field(
        default=METHOD_DFT,
        description="Method for geometry optimization.\n"
        "# If set to `null` and use xyz inputs, geometry optimization is skipped.\n"
        f"# Supported methods: {', '.join(SUPPORTED_GEOMETRY_OPT_METHODS)}. ")

    ## ASE related fields: used only when geometry_opt_method is 'aimnet2'
    ase_optimizer_name: str = Field(
        default=BFGS_OPTIMIZER,
        description=
        f"Name of the optimizer to use in ASE. Supported values: {', '.join(SUPPORTED_ASE_OPTIMIZERS)}"
    )
    ase_force_conv: float = Field(
        default=1e-5, description="Convergence threshold for forces in ASE")
    ase_max_cycle: int = Field(
        default=1000,
        description="Maximum number of optimization cycles allowed in ASE")

    ## PySCF related fields: used in geometry optimization when geometry_opt_method is 'dft',
    ## and in the property calculations
    pyscf_basis: str = Field(default=_DEFAULT_BASIS,
                             description="Basis set for PySCF calculations")
    pyscf_functional: str = Field(
        default=_DEFAULT_FUNCTIONAL,
        description="DFT functional for PySCF calculations")
    pyscf_max_scf_cycle: int = Field(
        default=_DEFAULT_SCF_MAX_CYCLE,
        description="Maximum number of SCF iterations allowed")
    pyscf_scf_conv_tol: float = Field(default=_DEFAULT_SCF_CONV_TOL,
                                      description="SCF convergence tolerance")
    pyscf_grids_level: int = Field(
        default=_DEFAULT_GRIDS_LEVEL,
        description="Level of grid refinement for numerical integration")
    # TODO: need to evaluate if this option is needed, if so, create separate module
    # for conformer sampling
    pyscf_save_fock: bool = Field(
        default=False, description="Whether to save the Fock matrix")

    ## Settings for ESP grid generation
    esp_solvent_accessible_region: float = Field(
        default=3.0,
        description="Solvent accessible region for ESP calculations in angstrom"
    )
    esp_grid_spacing: float = Field(
        default=0.5,
        description="Grid spacing for ESP calculations in angstrom")
    esp_probe_depth: float = Field(
        default=1.1,
        description="Probe depth for ESP calculations in angstrom")

    # Property calculation settings
    additional_properties: set[str] = Field(
        default=AdditionalProperty._default_props_solvent(),
        max_length=len(SUPPORTED_ADDITIONAL_PROPS),
        description="Additional DFT properties to compute.\n"
        f"# Supported properties: {', '.join(SUPPORTED_ADDITIONAL_PROPS)}\n"
        f"# Total electronic energy, HOMO/LUMO, dipole moment are always returned."
    )

    # Output settings
    output_dir: str = Field(
        default=Path.cwd().resolve(),
        description="Directory to save the output files.\n"
        "# Default to the current working directory where the cmdline unitlity is called."
    )

    output_file_format: Literal["csv", "parquet", "parq"] = Field(
        default="csv",
        description=
        "Output file format to write molecule-level and atom-level properties.\n"
        f"# Supported formats: {', '.join(SUPPORTED_COLUMNAR_FILE_FORMATS)}.")

    progress_log_interval: int = Field(
        default=10,
        description="Interval for logging progress during batch processing.")

    def get_property_kwargs(self) -> dict[str, bool]:
        """
        Convert additional_properties list to keyword arguments of `get_properties_main`

        :return: Dictionary mapping parameter names to boolean values.
        """
        prop_mapping = AdditionalProperty.to_kwargs_mapping()
        return {
            param_name: prop_name in self.additional_properties
            for prop_name, param_name in prop_mapping.items()
        }

    def to_ase_options(self) -> ASEOption:
        """
        Convert configuration settings to ASEOption.

        :return: An ASEOption object with settings from this configuration.
        """
        return ASEOption(optimizer_name=self.ase_optimizer_name,
                         force_conv=self.ase_force_conv,
                         max_cycle=self.ase_max_cycle)

    def to_pyscf_options(self) -> PySCFOption:
        """
        Convert configuration settings to PySCFOption.

        :return: A PySCFOption object with settings from this configuration.
        """
        return PySCFOption(basis=self.pyscf_basis,
                           dft_functional=self.pyscf_functional,
                           max_scf_cycle=self.pyscf_max_scf_cycle,
                           scf_conv_tol=self.pyscf_scf_conv_tol,
                           grids_level=self.pyscf_grids_level)

    def to_esp_grids_options(self) -> ESPGridsOption:
        """
        Convert configuration settings to ESPGridsOption.

        :return: An ESPGridsOption object with settings from this configuration.
        """
        return ESPGridsOption(
            solvent_accessible_region=self.esp_solvent_accessible_region,
            grid_spacing=self.esp_grid_spacing,
            probe_depth=self.esp_probe_depth)

    @classmethod
    def write_default_config_to_yaml(cls, yaml_path: str | Path) -> None:
        """
        Write the default config YAML file with descriptions as comments.
        """
        with open(yaml_path, "w") as file:
            file.write(cls._get_default_settings_yaml_string())

    @classmethod
    def _get_default_settings_yaml_string(cls) -> str:
        """
        Get the default settings as a YAML string with descriptions as comments.
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

        return "\n".join(yaml_lines)

    def to_recreate_string(self) -> str:
        """
        Return a string that can be copied and pasted to recreate this model.
        The returned string is valid Python code that can be executed to create
        an identical instance of the model.
        """
        lines = ["PipelineSettings("]
        for key, value in self.model_dump().items():
            lines.append(f"    {key}={repr(value)},")
        lines.append(")")

        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineSettings":
        """
        Create a PipelineSettings instance from a YAML file.

        :param yaml_path: Path to the YAML configuration file.
        :return: A PipelineSettings instance.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as fhandle:
            config_dict = yaml.safe_load(fhandle)
        # Validate and parse using the Pydantic model
        return cls(**config_dict)

    @field_validator('input_file_or_dir')
    def validate_input_existence(cls, input_file_or_dir: str) -> str:
        # Note that `mqc_pipeline.validate.validate_input` function provides more
        # detailed validation for the input file or directory.
        # The logic is separeted to make the current model more flexible in setting
        # up batching jobs.
        if not Path(input_file_or_dir).exists():
            raise ValidationError(
                f"Input file or directory does not exist: {input_file_or_dir}")
        return str(input_file_or_dir)

    @field_validator('geometry_opt_method')
    def validate_geometry_opt_method_basic(cls,
                                           method: str | None) -> str | None:
        """
        Basic validation for geometry optimization method.
        """
        if method is not None and method.upper(
        ) not in SUPPORTED_GEOMETRY_OPT_METHODS:
            raise ValidationError(
                f"Unsupported geometry optimization method: {method}. "
                f"Supported methods are: {', '.join(SUPPORTED_GEOMETRY_OPT_METHODS)}"
            )
        return method

    @model_validator(mode='after')
    def validate_geometry_opt_with_input(self) -> 'PipelineSettings':
        """
        Cross-field validation for geometry optimization method and input type.
        """
        # xyz inputs are allowed to skip geometry optimization,
        # so method can be None.
        if (self.geometry_opt_method is None
                and not Path(self.input_file_or_dir).is_dir()):
            raise ValidationError(
                "Geometry optimization method cannot be None for SMILES inputs. "
                "Please specify a valid geometry optimization method.")
        return self

    @field_validator('ase_optimizer_name')
    def validate_ase_optimizer(cls, optimizer: str) -> str:
        """
        Validates that the ASE optimizer name is one of the supported values.
        """
        if optimizer not in SUPPORTED_ASE_OPTIMIZERS:
            raise ValidationError(
                f"Unsupported ASE optimizer: {optimizer}. Supported optimizers are: {', '.join(SUPPORTED_ASE_OPTIMIZERS)}"
            )
        return optimizer

    @field_validator('additional_properties')
    def validate_additional_properties(cls,
                                       properties: list[str]) -> list[str]:
        """
        Validates that all specified additional properties are supported.
        """
        if unsupported := [
                prop for prop in properties
                if prop.lower() not in SUPPORTED_ADDITIONAL_PROPS
        ]:
            raise ValidationError(
                f"Unsupported additional properties: {', '.join(unsupported)}. "
                f"Supported properties are: {', '.join(SUPPORTED_ADDITIONAL_PROPS)}"
            )
        return properties
