import subprocess
import numpy as np
import pytest

from mqc_pipeline import optimize, Structure
from mqc_pipeline.property import DFT_FORCES_KEY


def has_nvidia_gpu():
    try:
        # Check if 'nvidia-smi' command is available and outputs something
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0 and len(result.stdout) > 0
    except FileNotFoundError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not has_nvidia_gpu(), reason="Test requires NVIDIA GPU")
def test_optimize_by_pyscf():
    xyz_before = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    st = Structure(elements=['H', 'H'],
                   xyz=xyz_before,
                   atomic_numbers=[1, 1],
                   unique_id='12345678902',
                   smiles='[H][H]')

    options = optimize.PySCFOption()
    st_opt = optimize.optimize_by_pyscf(st, options)
    assert not np.allclose(st_opt.xyz, xyz_before)

    for f_prop_key in f'{DFT_FORCES_KEY}_x', f'{DFT_FORCES_KEY}_y', f'{DFT_FORCES_KEY}_z':
        assert st_opt.atom_property.get(f_prop_key) is not None
