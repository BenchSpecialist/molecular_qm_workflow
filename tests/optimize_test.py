import numpy as np
import pytest

from mqc_pipeline import optimize, Structure


@pytest.mark.slow
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
    assert st_opt.property['dft_energy'] is not None
    assert st_opt.property['dft_forces'] is not None
