import yaml
from pathlib import Path

import pytest

from mqc_pipeline.settings import (PipelineSettings, ASEOption, PySCFOption,
                                   BFGS_OPTIMIZER, FIRE_OPTIMIZER,
                                   ValidationError)


def test_PySCFOption():
    pyscf_option = PySCFOption.get_default_anion_setting()
    assert pyscf_option.basis == "6311g*"
    assert pyscf_option.dft_functional == "b3lypg"
    assert pyscf_option.solvent_method == 'IEF-PCM'
    assert pyscf_option.solvent_eps == 18.5
    assert pyscf_option.dispersion == 'd3bj'


def test_minimal_initialization(tmp_cwd):
    inp_name = "smiles.txt"
    # Create the input file to pass the input validation
    Path(inp_name).touch()
    user_dict = {"input_file_or_dir": inp_name}
    PipelineSettings(**user_dict)


def test_validate_input_file_or_dir(tmp_cwd):
    # Raise error if the input file path cannot be found
    inp_name = "smiles.csv"
    with pytest.raises(ValidationError,
                       match="Input file or directory does not exist"):
        PipelineSettings(input_file_or_dir=inp_name)

    # Create the input file and check if the input validation passes
    Path(inp_name).write_text("smiles\nC\n")
    PipelineSettings(input_file_or_dir=inp_name)

    # Test with a directory
    xyz_dir = "input_xyz_dir"
    with pytest.raises(ValidationError,
                       match="Input file or directory does not exist:"):
        PipelineSettings(input_file_or_dir=xyz_dir)

    Path(xyz_dir).mkdir(exist_ok=True)
    # Create the directory and check if the validation passes
    PipelineSettings(input_file_or_dir=xyz_dir)


def test_validate_ase_optimizer(tmp_cwd):
    """Test that ASE optimizer validation works correctly."""
    Path("input.txt").write_text("C\n")
    user_dict = {"input_file_or_dir": "input.txt"}
    # Valid optimizers should not raise exceptions
    for optimizer in [BFGS_OPTIMIZER, FIRE_OPTIMIZER]:
        user_dict["ase_optimizer_name"] = optimizer
        PipelineSettings(**user_dict)

    # Invalid optimizer should raise ValueError
    with pytest.raises(ValidationError,
                       match='Unsupported ASE optimizer') as excinfo:
        user_dict["ase_optimizer_name"] = "INVALID_OPTIMIZER"
        PipelineSettings(**user_dict)


def test_to_ase_options(tmp_cwd):
    """Test conversion to ASEOption object."""
    Path("input.txt").write_text("C\n")
    user_input = {
        "input_file_or_dir": "input.txt",
        "ase_optimizer_name": FIRE_OPTIMIZER,
        "ase_force_conv": 1e-6,
        "ase_max_cycle": 2000,
    }

    config = PipelineSettings(**user_input)

    ase_options = config.to_ase_options()
    assert isinstance(ase_options, ASEOption)
    assert ase_options.optimizer_name == user_input["ase_optimizer_name"]
    assert ase_options.force_conv == user_input["ase_force_conv"]
    assert ase_options.max_cycle == user_input["ase_max_cycle"]


def test_to_pyscf_options(tmp_cwd):
    """Test conversion to PySCFOption object."""
    Path("input.txt").write_text("C\n")

    # Test basic settings without solvent (pyscf_solvent=False)
    user_input = {
        "input_file_or_dir": "input.txt",
        "pyscf_basis": "cc-pvdz",
        "pyscf_functional": "r2scan",
    }

    config = PipelineSettings(**user_input)

    pyscf_options = config.to_pyscf_options()
    assert isinstance(pyscf_options, PySCFOption)
    assert pyscf_options.basis == "cc-pvdz"
    assert pyscf_options.dft_functional == "r2scan"
    assert pyscf_options.solvent_method is None
    assert pyscf_options.solvent_eps is None
    assert pyscf_options.dispersion is None

    # Test with default solvent settings (pyscf_solvent=True)
    user_input["pyscf_solvent"] = True
    config = PipelineSettings(**user_input)
    pyscf_options = config.to_pyscf_options()
    assert pyscf_options.solvent_method == "IEF-PCM"
    assert pyscf_options.solvent_eps == 18.5

    # Test with custom solvent settings (tuple)
    user_input["pyscf_solvent"] = ("COSMO", 78.36)
    config = PipelineSettings(**user_input)
    pyscf_options = config.to_pyscf_options()
    assert pyscf_options.solvent_method == "COSMO"
    assert pyscf_options.solvent_eps == 78.36


def test_from_yaml(tmp_cwd):
    """Test creating a PipelineSettings instance from a YAML file."""
    # Create a temporary YAML file with custom settings
    Path("input.txt").write_text("C\n")
    custom_config = {
        'input_file_or_dir': "input.txt",
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
    # Other options take default values
    assert config.ase_max_cycle == PipelineSettings.model_fields[
        'ase_max_cycle'].default
    assert config.pyscf_functional == PipelineSettings.model_fields[
        'pyscf_functional'].default
    assert config.esp_probe_depth == PipelineSettings.model_fields[
        'esp_probe_depth'].default


def test_validate_additional_properties(tmp_cwd):
    """Test validation of additional properties."""
    Path("input.txt").write_text("C\n")
    user_input = {
        "input_file_or_dir": "input.txt",
        "additional_properties": ["chelpg_charges", "forces"]
    }
    # Valid additional properties should not raise exceptions
    PipelineSettings(**user_input)

    # Invalid additional property should raise ValidationError
    with pytest.raises(ValidationError,
                       match='Unsupported additional properties') as excinfo:
        user_input["additional_properties"] = ["forces", "INVALID_PROP"]
        PipelineSettings(**user_input)

    # Duplicates in the user input are removed
    props_with_duplicates = ["freq", "forces", "freq"]
    user_input["additional_properties"] = props_with_duplicates
    config = PipelineSettings(**user_input)
    assert config.additional_properties == set(props_with_duplicates)
