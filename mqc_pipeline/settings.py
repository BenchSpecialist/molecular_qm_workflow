from dataclasses import dataclass


@dataclass
class ASEOption:
    """
    Configuration options for ASE backend.

    :param optimizer_name: Name of the optimizer to use (e.g., 'BFGS', 'FIRE').
    :param force_conv: Convergence threshold for forces.
    :param max_cycle: Maximum number of optimization cycles allowed.
    """
    optimizer_name: str = 'BFGS'
    force_conv: float = 1e-5
    max_cycle: int = 1000


@dataclass
class PySCFOption:
    """
    Configuration options for PySCF backend.

    :param dft_functional: DFT functional to use.
    :param basis: Basis set to use.
    :param max_scf_cycle: Maximum number of SCF iterations allowed.
    :param scf_conv_tol: SCF convergence tolerance.
    :param grids_level: Level of grid refinement for numerical integration.
    """
    basis: str = '6311g*'
    dft_functional: str = 'b3lypg'
    max_scf_cycle: int = 100  # default: 50 in pyscf
    scf_conv_tol: float = 1e-09  # default: 1e-09 in pyscf
    grids_level: int = 3  # default: 3 in pyscf
