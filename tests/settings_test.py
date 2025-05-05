import os
import pytest
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from mqc_pipeline.settings import (PipelineSettings, ASEOption, PySCFOption,
                                   BFGS_OPTIMIZER, FIRE_OPTIMIZER,
                                   SUPPORTED_ASE_OPTIMIZERS)

default_config = {
    'geometry_opt_method': 'dft',
    'ase_optimizer_name': 'BFGS',
    'ase_force_conv': 1e-5,
    'ase_max_cycle': 1000,
    'pyscf_basis': "6311g*",
    'pyscf_functional': "b3lypg",
    'pyscf_max_scf_cycle': 500,
    'pyscf_scf_conv_tol': 1e-09,
    'pyscf_grids_level': 3,
    'pyscf_save_fock': False,
    'esp_solvent_accessible_region': 3.0,
    'esp_grid_spacing': 0.5,
    'esp_probe_depth': 1.1
}


def test_default_initialization():
    """Test that PipelineSettings initializes with default values."""
    config = PipelineSettings()

    # Convert the config to a dictionary and compare with default_config
    config_dict = config.model_dump()
    assert config_dict == default_config


def test_ase_optimizer_validation():
    """Test that ASE optimizer validation works correctly."""
    # Valid optimizers should not raise exceptions
    PipelineSettings(ase_optimizer_name=BFGS_OPTIMIZER)
    PipelineSettings(ase_optimizer_name=FIRE_OPTIMIZER)

    # Invalid optimizer should raise ValueError
    with pytest.raises(ValueError,
                       match='Unsupported ASE optimizer') as excinfo:
        PipelineSettings(ase_optimizer_name="INVALID_OPTIMIZER")

    # Check error message
    assert all(opt in str(excinfo.value) for opt in SUPPORTED_ASE_OPTIMIZERS)


def test_to_ase_options():
    """Test conversion to ASEOption object."""
    config = PipelineSettings(ase_optimizer_name=FIRE_OPTIMIZER,
                              ase_force_conv=1e-6,
                              ase_max_cycle=2000)

    ase_options = config.to_ase_options()

    assert isinstance(ase_options, ASEOption)
    assert ase_options.optimizer_name == FIRE_OPTIMIZER
    assert ase_options.force_conv == 1e-6
    assert ase_options.max_cycle == 2000


def test_to_pyscf_options():
    """Test conversion to PySCFOption object."""
    config = PipelineSettings(pyscf_basis="cc-pvdz", pyscf_functional="r2scan")

    pyscf_options = config.to_pyscf_options()
    assert isinstance(pyscf_options, PySCFOption)
    assert pyscf_options.basis == "cc-pvdz"
    assert pyscf_options.dft_functional == "r2scan"
    assert pyscf_options.max_scf_cycle == 500


def test_from_yaml(tmp_cwd):
    """Test creating a PipelineSettings instance from a YAML file."""
    # Create a temporary YAML file with custom settings
    custom_config = {
        'ase_optimizer_name': FIRE_OPTIMIZER,
        'pyscf_basis': 'sto-3g',
    }
    temp_yaml_file = "custom_config.yaml"
    with open(temp_yaml_file, "w") as f:
        yaml.dump(custom_config, f)

    # Create PipelineSettings instance from the YAML file
    config = PipelineSettings.from_yaml(temp_yaml_file)

    # Assert that the loaded config matches the custom settings
    assert config.ase_optimizer_name == FIRE_OPTIMIZER
    assert config.pyscf_basis == 'sto-3g'
    # Other options take default value
    assert config.ase_max_cycle == default_config['ase_max_cycle']
    assert config.pyscf_functional == default_config['pyscf_functional']
    assert config.esp_probe_depth == default_config['esp_probe_depth']
