import numpy as np
from functools import lru_cache

from ..constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV, VDW_RADII_ANGSTROM, BOHR
from ..import_util import import_cupy
from ..util import get_default_logger

logger = get_default_logger()

# If the absolute value of the ESP for closed-shell systems is greater than this
# threshold, the ESP calculation is considered to be unreliable.
# -2.0 to 2.0 Eh
_ESP_ABS_THRESHOLD_EV_SPIN0 = 55

SOLVENT_REGION_CUTOFF = 3.0  # in Angstrom
GRID_SPACING = 0.5  # in Angstrom
SOLVENT_PROBE = 1.1  # in Angstrom


@lru_cache(maxsize=1)
def import_int3c1e():
    try:
        from gpu4pyscf.gto import int3c1e
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Required GPU module 'gpu4pyscf.gto.int3c1e' not available: {str(e)}. "
            "Please ensure gpu4pyscf is installed correctly.")
    return int3c1e


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
             up to Z=56 (Barium).
    """
    prob_radius = solvent_probe * ANGSTROM_TO_BOHR
    esp_radii = ANGSTROM_TO_BOHR * np.array(VDW_RADII_ANGSTROM)
    esp_radii += prob_radius

    return esp_radii


def generate_esp_grids(mol,
                       rcut: float = SOLVENT_REGION_CUTOFF,
                       space: float = GRID_SPACING,
                       solvent_probe: float = SOLVENT_PROBE) -> np.ndarray:
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
    import importlib
    # Dynamic imports using importlib with error handling
    try:
        cupy = importlib.import_module('cupy')
        int3c2e = importlib.import_module('gpu4pyscf.df.int3c2e')
    except ImportError as e:
        raise RuntimeError(
            f"Required GPU modules not available: {str(e)}. Please ensure cupy and gpu4pyscf are installed."
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


def get_esp(pyscf_mol, one_rdm: np.ndarray, **kwargs) -> tuple:
    """
    Calculate electrostatic potential (ESP) for a molecule with the given grid points.

    :param pyscf_mol: 'pyscf.gto.mole.Mole' object containing atomic coordinates, nuclear charges,
                and basis set info
    :param one_rdm: One-electron reduced density matrix representing electron distribution
                    (shape: [Nbasis, Nbasis])

    Optional parameters (passed as kwargs):
    :param grids: Array of 3D coordinates where ESP will be evaluated (shape: [Ngrid, 3])
    :param rcut: Cut-off distance (in angstrom) for the solvent accessible region around the molecule (default: 3.0)
    :param space: Grid spacing (in angstrom) for the regularly spaced grid points (default: 0.5)
    :param solvent_probe: Radius (in angstrom) determining the envelope around the molecule (default: 1.1)

    :return: For neutral and cationic species, a 2-tuple containing minimum and maximum ESP values in eV.
             For anionic species, a 3-tuple containing minimum ESP, maximum ESP, and a MolBlock string
             representing the modified structure with Li atoms added at potential binding sites.
    """
    cupy, _CUPY_AVAILABLE = import_cupy()
    if not _CUPY_AVAILABLE:
        raise RuntimeError(
            "get_esp_range: Required CuPy package not available.")
    int3c1e = import_int3c1e()
    if 'grids' not in kwargs:
        grids = generate_esp_grids(pyscf_mol,
                                   rcut=kwargs.get('rcut',
                                                   SOLVENT_REGION_CUTOFF),
                                   space=kwargs.get('space', GRID_SPACING),
                                   solvent_probe=kwargs.get(
                                       'solvent_probe', SOLVENT_PROBE))
    # Calculate electronic contribution to ESP at grid points
    # This integral represents the Coulomb potential from the electron density
    # v_grids_e = ∫ ρ(r')/|r-r'| dr', where ρ is the electron density
    v_grids_e = int3c1e.int1e_grids(pyscf_mol, grids, dm=one_rdm)

    # Move calculations to GPU for better performance
    qm_xyz = cupy.array(pyscf_mol.atom_coords())
    qm_charges = cupy.array(pyscf_mol.atom_charges())
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
    if pyscf_mol.charge < 0:
        # For anions, also identify potential binding sites for Li+
        return get_esp_range_and_binding_site(pyscf_mol, grids, esp_vals)
    else:
        # For neutral and cationic species, just return the ESP range
        return get_esp_range(esp_vals)


def get_esp_range(esp_vals: np.ndarray) -> tuple[float, float]:
    """
    Determine the minimum and maximum electrostatic potential (ESP) values from
    the provided ESP values array.

    :param esp_vals: Array of ESP values at the specified grid points. For open-shell species,
                     this can be a 2D array with shape (2, Ngrid) where the
                     first row corresponds to alpha spin and the second to beta spin.
                     For closed-shell species, this is a 1D array with shape (Ngrid,).
    :return: Minimum and maximum ESP values across all grid points in eV unit.
    """
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
            logger.warning(
                f"ESP range (min: {esp_min:.2f} eV, max: {esp_max:.2f} eV) for "
                f"closed shell molecule is beyond the expected range (-{_ESP_ABS_THRESHOLD_EV_SPIN0} to {_ESP_ABS_THRESHOLD_EV_SPIN0} eV)."
            )
        return esp_min, esp_max


def get_esp_range_and_binding_site(
        pyscf_mol,
        grids: np.ndarray,
        esp_vals: np.ndarray,
        tolerance=0.01) -> tuple[float, float, str] | tuple[float, float]:
    """
    Identify potential binding sites for a Li+ ion based on the electrostatic potential (ESP)
    values calculated on a grid around the molecule. The function places a Li atom at each
    grid point where the ESP is within a specified tolerance of the minimum ESP value,
    and then applies a PCA-based rotation to minimize atomic occlusion. The resulting
    structure is returned in MolBlock format.

    :param pyscf_mol: PySCF Molecule object containing molecular information
    :param grids: Array of 3D coordinates where ESP was evaluated (shape: [Ngrid, 3])
    :param esp_vals: Array of ESP values at the specified grid points (shape: [Ngrid,])
    :param tolerance: Fractional tolerance around the minimum ESP value to consider
                      for placing Li atoms (default: 0.01, i.e., 1%)
    :return: Tuple containing minimum and maximum values of ESP in eV, also a
             MolBlock string representing the modified structure with Li atoms added
             if successful. If an error occurs, only the ESP min and max are returned.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D
    from sklearn.decomposition import PCA

    esp_min, esp_max = get_esp_range(esp_vals)
    # Find the points within `tolerance` of the ESP min
    tol_cutoff = esp_min + tolerance * abs(esp_min)
    near_min_points = grids[esp_vals <= tol_cutoff]

    try:
        # Create an editable RDKit molecule to work with
        editable_mol = Chem.RWMol(
            Chem.MolFromXYZBlock(pyscf_mol.tostring(format='xyz')))
        rdDetermineBonds.DetermineBonds(editable_mol, charge=pyscf_mol.charge)
        conf = editable_mol.GetConformer()

        # Place a Li atom at each near-min point
        for point in near_min_points:
            atom_idx = editable_mol.AddAtom(Chem.Atom('Li'))
            x, y, z = point * BOHR
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y),
                                                   float(z)))
            editable_mol.GetAtomWithIdx(atom_idx).SetFormalCharge(+1)

        # Run PCA to derive a rotation that minimizes atomic occlusion
        num_atoms = editable_mol.GetNumAtoms()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
        mol_center = coords.mean(axis=0)
        coords_centered = coords - mol_center
        pca = PCA(n_components=3)
        pca.fit(coords_centered)
        rot_matrix = pca.components_.T

        # Apply rotation and re-add molecule centroid
        rotated_coords = coords_centered @ rot_matrix + mol_center

        # Update the coordinates of the conformer
        for i in range(num_atoms):
            conf.SetAtomPosition(i, Point3D(*rotated_coords[i]))

        return esp_min, esp_max, Chem.MolToMolBlock(editable_mol)

    except Exception as e:
        logger.error(f"Error generating Li+ binding site: {str(e)}")
        return esp_min, esp_max
