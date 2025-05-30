import numpy as np
from pathlib import Path
from pyscf.tools import molden
from pyscf.dft import uks

from mqc_pipeline.property import stability

import pytest


@pytest.fixture(scope="session")
def mf_object_ec_anion():
    """
    Create a PySCF mean-field object for ethylene carbonate anion (open-shell) for testing.
    """
    mol, mo_energy, _, mo_occ, _, _ = molden.load(
        Path(__file__).parent / 'data' / 'ethylene_carbonate_anion.molden')

    mf_object = uks.UKS(mol, xc='b3lypg').density_fit()
    mf_object.mo_energy = np.array(mo_energy)
    mf_object.mo_occ = np.array(mo_occ)
    return mf_object


def test_get_homo_lumo_levels(mf_object_ec_anion):
    homo_ev, lumo_ev = stability.get_homo_lumo_levels(mf_object_ec_anion)
    assert np.isclose(homo_ev, -2.333548)
    assert np.isclose(lumo_ev, 1.169537)


def test_get_open_shell_averaged_levels(mf_object_ec_anion):
    homo_level_au, lumo_level_au = stability._get_open_shell_averaged_levels(
        mf_object_ec_anion)
    assert np.isclose(homo_level_au, -4.057808)
    assert np.isclose(lumo_level_au, 1.538804)


def test_get_open_shell_effective_levels(mf_object_ec_anion):
    homo_level_au, lumo_level_au = stability._get_open_shell_effective_levels(
        mf_object_ec_anion)
    assert np.isclose(homo_level_au, -2.333548)
    assert np.isclose(lumo_level_au, 2.014434)
