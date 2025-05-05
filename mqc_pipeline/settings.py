import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass

### Module constants ###
METHOD_AIMNet2 = 'aimnet2'
METHOD_DFT = 'dft'
SUPPORTED_GEOMETRY_OPT_METHODS = [METHOD_AIMNet2, METHOD_DFT]

BFGS_OPTIMIZER = "BFGS"
FIRE_OPTIMIZER = "FIRE"
SUPPORTED_ASE_OPTIMIZERS = (BFGS_OPTIMIZER, FIRE_OPTIMIZER)


@dataclass
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


@dataclass
class PySCFOption:
    """
    Configuration options for PySCF backend.

    :param dft_functional: DFT functional to use.
    :param basis: Basis set to use.
    :param max_scf_cycle: Maximum number of SCF iterations allowed.
    :param scf_conv_tol: SCF convergence tolerance.
    :param grids_level: Level of grid refinement for numerical integration.
    """
    basis: str = '6311g*'
    dft_functional: str = 'b3lypg'
    max_scf_cycle: int = 100  # default: 50 in pyscf
    scf_conv_tol: float = 1e-09  # default: 1e-09 in pyscf
    grids_level: int = 3  # default: 3 in pyscf


class PipelineSettings(BaseModel):
    """
    Frontend schema for the pipeline settings.
    """
    # Required settings
    geometry_opt_method: str = Field(
        default="dft", description="Method for geometry optimization.")

    @field_validator('geometry_opt_method')
    def validate_geometry_opt_method(cls, method: str) -> str:
        if method.lower() not in SUPPORTED_GEOMETRY_OPT_METHODS:
            raise ValueError(
                f"Unsupported geometry optimization method: {method}. Supported methods are: {', '.join(SUPPORTED_GEOMETRY_OPT_METHODS)}"
            )
        return method

    # Optional settings
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
    pyscf_basis: str = Field(default="6311g*",
                             description="Basis set for PySCF calculations")
    pyscf_functional: str = Field(
        default="b3lypg", description="DFT functional for PySCF calculations")
    pyscf_max_scf_cycle: int = Field(
        default=500, description="Maximum number of SCF iterations allowed")
    pyscf_scf_conv_tol: float = Field(default=1e-09,
                                      description="SCF convergence tolerance")
    pyscf_grids_level: int = Field(
        default=3,
        description="Level of grid refinement for numerical integration")
    pyscf_save_fock: bool = Field(
        default=False, description="Whether to save the Fock matrix")

    ## Settings for ESP grid generation
    esp_solvent_accessible_region: float = Field(
        default=3.0,
        description="Solvent accessible region for ESP calculations in Å")
    esp_grid_spacing: float = Field(
        default=0.5, description="Grid spacing for ESP calculations in Å")
    esp_probe_depth: float = Field(
        default=1.1, description="Probe depth for ESP calculations in Å")

    @field_validator('ase_optimizer_name')
    def validate_ase_optimizer(cls, optimizer: str) -> str:
        """
        Validates that the ASE optimizer name is one of the supported values.

        :param v: The optimizer name to validate.
        :return: The validated optimizer name.
        :raises ValueError: If the optimizer name is not supported.
        """
        if optimizer not in SUPPORTED_ASE_OPTIMIZERS:
            raise ValueError(
                f"Unsupported ASE optimizer: {optimizer}. Supported optimizers are: {', '.join(SUPPORTED_ASE_OPTIMIZERS)}"
            )
        return optimizer

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

    @classmethod
    def write_default_config_to_yaml(cls, yaml_path: str | Path) -> None:
        """
        Write the default config YAML file.
        """
        with open(yaml_path, "w") as file:
            file.write(cls._get_default_settings_yaml_string())

    @classmethod
    def _get_default_settings_yaml_string(cls):
        """
        Get the default settings to a YAML string.
        """
        schema = cls.schema()
        default_strings = []
        for key, val_dict in schema["properties"].items():
            default_value = val_dict.get("default", ' ')
            default_strings.append(f"{key}: {default_value}")

        return "\n".join(default_strings)

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
