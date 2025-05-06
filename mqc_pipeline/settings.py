import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass

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
    """
    # Required settings
    geometry_opt_method: str = Field(
        default=METHOD_DFT, description="Method for geometry optimization.")

    @field_validator('geometry_opt_method')
    def validate_geometry_opt_method(cls, method: str) -> str:
        if method.upper() not in SUPPORTED_GEOMETRY_OPT_METHODS:
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
