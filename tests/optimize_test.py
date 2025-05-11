import platform
import numpy as np
import pytest

from mqc_pipeline import optimize, Structure
from mqc_pipeline.property import DFT_ENERGY_KEY, DFT_FORCES_KEY
from mqc_pipeline.util import has_nvidia_gpu


@pytest.mark.slow
@pytest.mark.skipif(not has_nvidia_gpu(), reason="Test requires NVIDIA GPU")
@pytest.mark.skipif(platform.system() != "Linux", reason="Only runs on Linux")
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
    assert st_opt.property[DFT_ENERGY_KEY] is not None
    assert st_opt.property[DFT_FORCES_KEY] is not None
