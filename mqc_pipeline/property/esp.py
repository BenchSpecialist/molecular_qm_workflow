import importlib
import numpy as np
import pyscf

from ..constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV


def _get_esp_radii(solvent_probe: float) -> np.ndarray:
    """
    Generate atomic van der Waals radii for ESP calculations with solvent effects.
    This generates an array of atomic radii (in Bohr) that define the boundary
    between the solvent-accessible and solvent-inaccessible regions around a molecule.
    The base radii values are standard van der Waals radii for each element which are then
    increased by the probe radius to account for the size of the solvent molecule.

    :param solvent_probe: radius of the solvent probe molecule in Angstroms.
                          This value is added to each atomic radius to determine
                          the solvent-inaccessible boundary.

    :return: array of modified van der Waals radii in Bohr units for elements
             up to Z=56 (Barium). The first element (index 0) represents a ghost atom.
             Each subsequent index corresponds to the atomic number Z.
    """
    esp_radii = ANGSTROM_TO_BOHR * np.array(
        [0,                                              # Ghost atom
         0.30,                                     1.22, # 1s
         1.23, 0.89, 0.88, 0.77, 0.70, 0.66, 0.58, 1.60, # 2s2p
         1.40, 1.36, 1.25, 1.17, 1.10, 1.04, 0.99, 1.91, # 3s3p
         2.03, 1.74,                                     # 4s (K, Ca)
         1.44, 1.32, 1.22, 1.19, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25, # 3d (Sc,.., Zn)
         1.25, 1.22, 1.21, 1.17, 1.14, 1.98,             # 4p (Ga, .., Kr)
         2.22, 1.92,                                     # 5s (Rb, Sr)
         1.62, 1.45, 1.34, 1.29, 1.27, 1.24, 1.25, 1.28, 1.34, 1.41, # 4d (Y,..,Cd)
         1.50, 1.40, 1.41, 1.37, 1.33, 2.09,             # 5p (In,.., Xe)
         2.35, 1.98]                                     # 6s
    )  # yapf: disable

    prob_radius = solvent_probe * ANGSTROM_TO_BOHR
    esp_radii += prob_radius

    return esp_radii


def generate_esp_grids(mol: 'pyscf.gto.mole.Mole',
                       rcut: float = 3.0,
                       space: float = 0.5,
                       solvent_probe: float = 0.7) -> np.ndarray:
    """
    Generate grid points for electrostatic potential (ESP) calculation

    :param mol: PySCF Mole object providing coordinates and atomic numbers information.
    :param rcut: cut-off distance (in angstrom) for the solvent accessible region around the molecule
    :param space: grid spacing (in angstrom) for the regularly spaced grid points
    :param solvent_probe: radius (in angstrom) determining the envelope around the molecule
    """
    esp_radii = _get_esp_radii(solvent_probe)
    qm_xyz = mol.atom_coords()
    qm_znum = mol.atom_charges()
    natom = qm_znum.shape[0]

    grid_min = np.array(
        [min(qm_xyz[:, 0]),
         min(qm_xyz[:, 1]),
         min(qm_xyz[:, 2])])
    grid_max = np.array(
        [max(qm_xyz[:, 0]),
         max(qm_xyz[:, 1]),
         max(qm_xyz[:, 2])])

    ## calculate the grid size
    _rcut = rcut * ANGSTROM_TO_BOHR
    _rcut2 = _rcut * _rcut
    _spac = space * ANGSTROM_TO_BOHR

    ngrid_x = int((grid_max[0] - grid_min[0] + 2.0 * _rcut) / _spac) + 1
    ngrid_y = int((grid_max[1] - grid_min[1] + 2.0 * _rcut) / _spac) + 1
    ngrid_z = int((grid_max[2] - grid_min[2] + 2.0 * _rcut) / _spac) + 1

    small = 1.0e-8
    grids = []

    for iz in range(ngrid_z):
        for iy in range(ngrid_y):
            for ix in range(ngrid_x):
                gv = grid_min - _rcut + _spac * np.array([ix, iy, iz])

                rmin2 = _rcut2
                lupdate = True

                for ia in range(natom):
                    znum = qm_znum[ia]
                    rad = esp_radii[znum]
                    rad2 = rad * rad

                    dr = gv - qm_xyz[ia]
                    r2 = np.einsum('i,i', dr, dr)

                    if rad2 - r2 > small:
                        lupdate = False
                        break

                    if rmin2 - r2 > small:
                        rmin2 = r2

                if lupdate:
                    if _rcut2 - rmin2 > small:
                        grids.append(list(gv))

    grids = np.array(grids)

    return grids


def get_esp_range(mol: 'pyscf.gto.mole.Mole', grids: np.ndarray,
                  one_rdm: np.ndarray) -> tuple[float, float]:
    """
    Calculate electrostatic potential (ESP) for a molecule with the given grid points.

    :param mol: PySCF Mole object containing the molecule information.
    :param grids: numpy array representing grid points where ESP to be calculated.
                  shape (ngrid, 3)
    :param one_rdm: numpy array representing the one-body reduced density matrix.

    :return: a tuple containing the minimum and maximum ESP values in eV.
    """
    # Dynamic imports using importlib with error handling
    try:
        cupy = importlib.import_module('cupy')
        int3c2e = importlib.import_module('gpu4pyscf.df.int3c2e')
    except ImportError as e:
        raise RuntimeError(
            f"Required GPU modules not available: {e}. Please ensure cupy and gpu4pyscf are installed."
        )

    # Convert input data to CuPy arrays for GPU acceleration
    qm_xyz = cupy.array(mol.atom_coords())
    qm_charges = cupy.array(mol.atom_charges())
    grids = cupy.array(grids)

    # Calculate distances
    drg = qm_xyz[:, None, :] - grids  # (Natom, Ngrid, 3)
    dr = cupy.linalg.norm(drg, axis=2)  # (Natom, Ngrid)

    # Calculate z_val
    z_val = cupy.einsum('ig, i->g', 1.0 / dr, qm_charges)
    g_val = []

    # Calculate g_val using GPU4PySCF integrals
    auxmol = pyscf.gto.fakemol_for_charges(
        cupy.asnumpy(grids))  # Convert grids to numpy for PySCF compatibility
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=False, aosym=True, group_size=256)
    v_grids_e = 2.0 * int3c2e.get_j_int3c2e_pass1(
        intopt, cupy.asarray(one_rdm), sort_j=False)
    g_val = -v_grids_e

    grids_val = cupy.asnumpy(g_val + z_val)

    esp_max = max(grids_val) * HARTREE_TO_EV
    esp_min = min(grids_val) * HARTREE_TO_EV

    return esp_min, esp_max


def _get_esp_range(mol: 'pyscf.gto.mole.Mole', grids: np.ndarray,
                   one_rdm: np.ndarray) -> tuple[float, float]:
    """
    This function uses the handy `int1e_grids` function available in the latest
    GPU4PySCF version to compute the electrostatic potential with given grid points.
    """
    # Dynamic import of the int1e_grids function
    int1e_grids = importlib.import_module('gpu4pyscf.gto.int3c1e').int1e_grids

    esp_vals = int1e_grids(mol, grids, dm=one_rdm)
    return min(esp_vals) * HARTREE_TO_EV, max(esp_vals) * HARTREE_TO_EV
