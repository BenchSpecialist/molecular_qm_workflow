import os
import tempfile
import pytest
from mqc_pipeline import Structure


@pytest.fixture(autouse=True)
def tmp_cwd():
    """
    Change to a temporary directory for the duration of the test
    """
    old_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)
    yield
    os.chdir(old_cwd)


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
                         'energy': -40.528,
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
                         'energy': -109.556,
                         'charges': [0.0, 0.0],
                     })
