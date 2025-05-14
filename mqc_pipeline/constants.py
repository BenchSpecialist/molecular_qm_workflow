from scipy.constants import physical_constants
from pyscf.data.nist import BOHR

HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
EV_TO_HARTREE = 1 / HARTREE_TO_EV

# unit of `BOHR` is angstroms per Bohr
ANGSTROM_TO_BOHR = 1.0 / BOHR
