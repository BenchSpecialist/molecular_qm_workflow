import os
import shutil
import tempfile
import pytest
from mqc_pipeline import Structure
from mqc_pipeline.property import DFT_ENERGY_KEY, HOMO_KEY, LUMO_KEY


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
