import importlib
import numpy as np

from ..util import logger

# Dynamic imports of GPU modules
# This is done to avoid import errors when running on CPU-only systems
try:
    cupy = importlib.import_module('cupy')
    int1e_grids = importlib.import_module('gpu4pyscf.gto.int3c1e').int1e_grids
    _USE_GPU = True
except (ImportError, AttributeError) as e:
    _USE_GPU = False

from ..constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV

# If the absolute value of the ESP for closed-shell systems is greater than this
# threshold, the ESP calculation is considered to be unreliable.
_ESP_ABS_THRESHOLD_EV_SPIN0 = 27


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


def generate_esp_grids(mol,
                       rcut: float = 3.0,
                       space: float = 0.5,
                       solvent_probe: float = 1.1) -> np.ndarray:
    """
    Generate grid points for electrostatic potential (ESP) calculation

    :param mol: 'pyscf.gto.mole.Mole' object providing coordinates and atomic numbers information.
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


def _get_esp_range_old(mol, grids: np.ndarray,
                       one_rdm: np.ndarray) -> tuple[float, float]:
    """
    Old version of the ESP calculation using GPU4PySCF.
    It only works correct up to `gpu4pyscf 1.3.2`; for the latest
    version `gpu4pyscf 1.4.0`, this function output values that are well
    outside the range of (-1, 1).
    """
    import pyscf
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


def get_esp_range(mol, grids: np.ndarray,
                  one_rdm: np.ndarray) -> tuple[float, float]:
    """
    Calculate electrostatic potential (ESP) for a molecule with the given grid points.

    :param mol: 'pyscf.gto.mole.Mole' object containing atomic coordinates, nuclear charges,
                and basis set info
    :param grids: Array of 3D coordinates where ESP will be evaluated (shape: [Ngrid, 3])
    :param one_rdm: One-electron reduced density matrix representing electron distribution
                    (shape: [Nbasis, Nbasis])

    :return: Minimum and maximum ESP values across all grid points in eV unit.
    """
    if not _USE_GPU:
        raise RuntimeError(
            "Required GPU modules not available. Please ensure cupy and gpu4pyscf are installed correctly."
        )
    # Calculate electronic contribution to ESP at grid points
    # This integral represents the Coulomb potential from the electron density
    # v_grids_e = ∫ ρ(r')/|r-r'| dr', where ρ is the electron density
    v_grids_e = int1e_grids(mol, grids, dm=one_rdm)

    # Move calculations to GPU for better performance
    qm_xyz = cupy.array(mol.atom_coords())
    qm_charges = cupy.array(mol.atom_charges())
    grids_cu = cupy.array(grids)

    # Calculate distances between each nucleus and each grid point
    drg = qm_xyz[:, None, :] - grids_cu  # (Natom, Ngrid, 3)
    dr = cupy.linalg.norm(drg, axis=2)  # (Natom, Ngrid)

    # Calculate nuclear contribution to ESP
    # The ESP from nuclei is given by: Σ(Z_A/|r-R_A|) for all atoms A
    # where Z_A is the nuclear charge and R_A is the nuclear position
    z_val = cupy.einsum('ig, i->g', 1.0 / dr, qm_charges)

    # Combine nuclear and electronic contributions to get total ESP
    # and convert back to CPU memory (numpy array)
    # Electronic contribution has opposite sign (negative charges)
    # Total ESP = nuclear contribution - electronic contribution
    esp_vals = cupy.asnumpy(z_val - v_grids_e)

    # For open shell species, esp_vals has shape (2, Ngrid) where esp_vals[0] is
    # alpha ESP and esp_vals[1] is beta ESP.
    # Total ESP at any point is the contribution from all electrons, regardless of spin.
    if esp_vals.ndim == 2:
        # Sum the alpha and beta contributions at each grid point
        total_esp = esp_vals[0] + esp_vals[1]
        return min(total_esp) * HARTREE_TO_EV, max(total_esp) * HARTREE_TO_EV
    else:
        # For closed shell species, return the min and max directly
        esp_min = min(esp_vals) * HARTREE_TO_EV
        esp_max = max(esp_vals) * HARTREE_TO_EV
        if abs(esp_min) > _ESP_ABS_THRESHOLD_EV_SPIN0 or abs(
                esp_max) > _ESP_ABS_THRESHOLD_EV_SPIN0:
            raise ValueError(
                f"ESP range (min: {esp_min:.2f} eV, max: {esp_max:.2f} eV) for "
                "closed shell species are outside the expected range (-27 to 27 eV). "
            )
        return esp_min, esp_max
