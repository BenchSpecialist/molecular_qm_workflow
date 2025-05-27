from scipy.constants import physical_constants
from pyscf.data.nist import BOHR
from ase.data import chemical_symbols

HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
EV_TO_HARTREE = 1 / HARTREE_TO_EV

# unit of `BOHR` is angstroms per Bohr
ANGSTROM_TO_BOHR = 1.0 / BOHR

# Standard Van der Waals radii in Angstrom
# Indexes 1-56 (atomic number Z) correspond to elements from Hydrogen (H, Z=1)
# to Barium (Ba, Z=56).
VDW_RADII_ANGSTROM = (
    0,                                              # Ghost atom
    0.30,                                     1.22, # 1s (H, He)
    1.23, 0.89, 0.88, 0.77, 0.70, 0.66, 0.58, 1.60, # 2s2p
    1.40, 1.36, 1.25, 1.17, 1.10, 1.04, 0.99, 1.91, # 3s3p
    2.03, 1.74,                                     # 4s (K, Ca)
    1.44, 1.32, 1.22, 1.19, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25, # 3d (Sc,.., Zn)
    1.25, 1.22, 1.21, 1.17, 1.14, 1.98,             # 4p (Ga, .., Kr)
    2.22, 1.92,                                     # 5s (Rb, Sr)
    1.62, 1.45, 1.34, 1.29, 1.27, 1.24, 1.25, 1.28, 1.34, 1.41, # 4d (Y,..,Cd)
    1.50, 1.40, 1.41, 1.37, 1.33, 2.09,             # 5p (In,.., Xe)
    2.35, 1.98                                      # 6s
)  # yapf: disable

# Chemical symbols to atomic numbers
ELEMENT_TO_ATOMIC_NUMBER = {
    symbol: i
    for i, symbol in enumerate(chemical_symbols)
}
