import os
import shutil
import tempfile
import platform
from typing import Callable

import pytest

from mqc_pipeline import Structure
from mqc_pipeline.property import DFT_ENERGY_KEY, HOMO_KEY, LUMO_KEY

# Platform detection constants
IS_DARWIN: bool = platform.system() == "Darwin"
IS_LINUX: bool = platform.system() == "Linux"
IS_WINDOWS: bool = platform.system() == "Windows"


# Decorator factories with optional custom reasons
def darwin_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Darwin platforms.
    """
    return pytest.mark.skipif(not IS_DARWIN,
                              reason=reason
                              or "Test only runs on macOS/Darwin")


def linux_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Linux platforms.
    """
    return pytest.mark.skipif(not IS_LINUX,
                              reason=reason or "Test only runs on Linux")


def windows_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Windows platforms.
    """
    return pytest.mark.skipif(not IS_WINDOWS,
                              reason=reason or "Test only runs on Windows")


def requires_openbabel(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests when OpenBabel pybel import fails.
    The GitHub unittests workflow cannot install openbabel correctly with pyproject.toml,
    so this decorator is used to skip tests when the package is not available.
    """
    try:
        from openbabel import pybel
        skip_condition = False
    except ImportError:
        skip_condition = True

    return pytest.mark.skipif(skip_condition,
                              reason=reason or "OpenBabel not available")


@pytest.fixture(autouse=True)
def tmp_cwd():
    """
    Change to a temporary directory for the duration of the test
    """
    old_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp_dir)


@pytest.fixture(scope="session")
def methane_st():
    return Structure(elements=['C', 'H', 'H', 'H', 'H'],
                     xyz=[[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                          [-0.629, -0.629, 0.629], [0.629, -0.629, -0.629],
                          [-0.629, 0.629, -0.629]],
                     atomic_numbers=[6, 1, 1, 1, 1],
                     unique_id='12345678901',
                     smiles='C',
                     property={
                         DFT_ENERGY_KEY: -40.528,
                         HOMO_KEY: -10.776,
                         LUMO_KEY: 1.442,
                     },
                     atom_property={
                         'charges': [0.0, 0.0, 0.0, 0.0, 0.0],
                     })


@pytest.fixture(scope="session")
def n2_st():
    return Structure(elements=['N', 'N'],
                     xyz=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
                     atomic_numbers=[7, 7],
                     unique_id='12345678902',
                     smiles='N#N',
                     property={
                         DFT_ENERGY_KEY: -109.556,
                         HOMO_KEY: -11.737,
                         LUMO_KEY: -0.715,
                     },
                     atom_property={
                         'charges': [0.0, 0.0],
                     })
