import numpy as np

from ..constants import VDW_RADII_ANGSTROM
from ..import_util import import_cupy
from .esp import generate_esp_grids

cupy, _CUPY_AVAILABLE = import_cupy()

try:
    from gpu4pyscf.gto.int3c1e import int1e_grids
    from gpu4pyscf.dft import numint
    _GPU4PYSCF_AVAILABLE = True
except (ImportError, AttributeError) as e:
    _GPU4PYSCF_AVAILABLE = False


def get_vdw_volume(coords_angstrom: np.ndarray,
                   atomic_numbers: np.ndarray | list[int],
                   n_samples: int = 1000000,
                   extra_padding: float = 0.001,
                   use_gpu: bool = False) -> float:
    """
    Estimate molecular volume using Van der Waals radii

    :param coords_angstrom: Coordinates of atoms in Angstroms (n_atoms, 3)
    :param atomic_numbers: Atomic numbers of atoms (n_atoms,)
    :param n_samples: Number of random points used in Monte Carlo integration
                      for volume estimation
    :param extra_padding: Additional padding beyond max vdW radius
    :param use_gpu: Whether to use GPU for Monte Carlo integration (requires CuPy)

    :return: Estimated molecular volume in Angstrom³
    """
    if use_gpu and (not _CUPY_AVAILABLE):
        raise RuntimeError(
            "get_vdw_volume: CuPy is required for GPU vdW volume calculation.")

    # Get radii for all atoms
    radii = [VDW_RADII_ANGSTROM[z] for z in atomic_numbers]

    # Bounding box
    padding = np.max(radii) + extra_padding
    min_coords = np.min(coords_angstrom, axis=0) - padding
    max_coords = np.max(coords_angstrom, axis=0) + padding
    box_volume = float(np.prod(max_coords - min_coords))

    n_samples = int(n_samples)
    if use_gpu:
        points_inside = _monte_carlo_gpu(coords_angstrom, radii, min_coords,
                                         max_coords, n_samples)
    else:
        points_inside = _monte_carlo_cpu(coords_angstrom,
                                         radii,
                                         min_coords,
                                         max_coords,
                                         n_samples,
                                         vectorized=True)

    # Volume is proportional to the fraction of points inside the spheres
    volume = box_volume * points_inside / n_samples
    return float(volume)


def _monte_carlo_gpu(coords_angstrom, radii, min_coords, max_coords,
                     n_samples) -> float:
    # Monte Carlo integration on GPU
    def _check_points_in_spheres(points, centers, radii) -> cupy.ndarray:
        """
        Return boolean array indicating which points are inside any sphere
        """
        # Calculate squared distances between all points and all centers by broadcasting
        diff = cupy.expand_dims(points, 1) - cupy.expand_dims(centers, 0)
        squared_distances = cupy.sum(diff**2, axis=2)
        squared_radii = radii**2
        # Check if each point is inside any sphere
        inside_any = cupy.any(squared_distances <= squared_radii, axis=1)
        return inside_any

    coords_gpu = cupy.array(coords_angstrom)
    radii_gpu = cupy.array(radii)
    min_coords_gpu = cupy.array(min_coords)
    max_coords_gpu = cupy.array(max_coords)

    points_gpu = cupy.random.uniform(min_coords_gpu,
                                     max_coords_gpu,
                                     size=(n_samples, 3))

    points_inside = cupy.sum(
        _check_points_in_spheres(points_gpu, coords_gpu, radii_gpu))

    return float(points_inside.get())


def _monte_carlo_cpu(coords_angstrom: np.ndarray,
                     radii: list[float],
                     min_coords: np.ndarray,
                     max_coords: np.ndarray,
                     n_samples: int,
                     vectorized: bool = True,
                     batch_size: int = 10000) -> float:
    """
    Run monte carlo integration to estimate the number of points inside the
    molecular volume defined by Van der Waals radii.

    :param coords_angstrom: Atomic coordinates in Angstroms
    :param radii: Van der Waals radii for each atom
    :param min_coords: Minimum coordinates of bounding box
    :param max_coords: Maximum coordinates of bounding box
    :param n_samples: Total number of random points
    :param vectorized: Whether to vectorize Monte Carlo using NumPy broadcasting
                      (single process)
    :param batch_size: Number of points to process in each batch

    :return: Number of points inside molecular volume
    """
    points_inside = 0

    if not vectorized:

        def point_in_any_sphere(point, centers, radii):
            for center, radius in zip(centers, radii):
                if np.linalg.norm(point - center) <= radius:
                    return True
            return False

        for _ in range(n_samples):
            point = np.random.uniform(min_coords, max_coords)
            if point_in_any_sphere(point, coords_angstrom, radii):
                points_inside += 1
        return float(points_inside)

    radii_array = np.array(radii)
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)

        # Generate batch of random points
        points = np.random.uniform(min_coords,
                                   max_coords,
                                   size=(current_batch, 3))

        # Calculate distances to all atoms for all points
        diff = points[:, np.newaxis, :] - coords_angstrom[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # Check if any point is inside any sphere
        inside_any = np.any(distances <= radii_array, axis=1)
        points_inside += np.sum(inside_any)

    return float(points_inside)


# TODO: fix this function
def _get_density_isosurface_volume(pyscf_mol,
                                   one_rdm,
                                   density_cutoff=0.001,
                                   padding=None,
                                   grid_spacing=0.2) -> float:
    """
    Calculate molecular volume from electron density isosurface. The value of
    padding used to set grid boundaries should be large enough to ensure the
    density isosurface is fully contained within the grid for the given density cutoff.
    Some empirical rules of thumb for padding:
    - rho = 0.01 au, 2-4 angstrom; rho = 0.001 au, 4-6 angstrom; rho = 0.0001 au, 6-8 angstrom
    - molecules with <10 atoms: 3-5 angstrom; > 50 atoms: 6-10 angstrom;
    - diffuse basis sets need larger padding;
    - charged systems may have longer-range density tails.

    :param pyscf_mol: PySCF Mole object containing molecular information
    :param one_rdm: One-body reduced density matrix (n_atoms, n_atoms)
    :param density_cutoff: Density threshold for isosurface
    :param padding: Padding around the molecule in angstroms
                    If None, adaptive padding is calculated based on molecular size
    :param grid_spacing: Spacing between grid points in angstroms
    """
    if not _GPU4PYSCF_AVAILABLE:
        raise RuntimeError(
            "get_density_isosurface_volume: GPU4PySCF is required for density isosurface volume calculation."
        )

    coords = pyscf_mol.atom_coords()  # (n_atoms, 3)

    if padding is None:
        # Adaptive padding based on molecular size and density cutoff
        mol_size = np.max(coords.max(axis=0) - coords.min(axis=0))

        # Electron density decays as exp(-αr) where α depends on the orbital
        # For ρ = 0.001, need r such that exp(-αr) = 0.001
        # This gives r ≈ 7/α, where α ≈ 1-2 Bohr⁻¹ for valence orbitals
        # Lower density threshold needs more more space/padding.

        # At least 3 Å or 30% of molecule size
        base_padding = max(3.0, mol_size * 0.3)
        # Scale with -log(threshold)
        threshold_factor = max(1.0, -np.log(density_cutoff) * 0.5)
        padding = base_padding * threshold_factor

        print(
            f"Using adaptive padding: {padding:.2f} Å (mol_size: {mol_size:.2f} Å, density cutoff: {density_cutoff})"
        )

    # Determine grid boundaries with padding
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding

    nx = int((max_coords[0] - min_coords[0]) / grid_spacing)
    ny = int((max_coords[1] - min_coords[1]) / grid_spacing)
    nz = int((max_coords[2] - min_coords[2]) / grid_spacing)

    # Create grid points
    x = np.linspace(min_coords[0], max_coords[0], nx)
    y = np.linspace(min_coords[1], max_coords[1], ny)
    z = np.linspace(min_coords[2], max_coords[2], nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    grid_coords = generate_esp_grids(pyscf_mol,
                                     rcut=padding,
                                     space=grid_spacing,
                                     solvent_probe=0.0)
    # Calculate electron density at grid points
    rho = int1e_grids(pyscf_mol, grid_coords, dm=one_rdm)

    # Evaluate density on grid
    # grid_coords_bohr = grid_coords * ANGSTROM_TO_BOHR
    # ao_value = numint.eval_ao(pyscf_mol, grid_coords_bohr.T)
    # rho = numint.eval_rho(pyscf_mol, ao_value, one_rdm)

    # Count points above density cutoff
    points_inside = np.sum(rho > density_cutoff)

    # Volume per grid point
    volume_per_point = grid_spacing**3
    total_volume = points_inside * volume_per_point

    return total_volume
