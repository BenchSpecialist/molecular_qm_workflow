import numpy as np
from uuid import uuid4
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from ase import Atoms
import pyscf

from .constants import DFT_FORCES_KEY

COORDINATE_UNIT = 'angstrom'
_DEFAULT_CHARGE = 0.
_DEFAULT_MULTIPLICITY = 1.
_UNIQUE_KEY_LENGTH = 11


@dataclass(slots=True)  # eliminates the __dict__ to reduce memory overhead
class Structure:
    """
    Container to hold information of a molecule.
    """
    elements: list[str]
    xyz: np.ndarray  # shape (n_atoms, 3)
    atomic_numbers: Optional[list[int]] = None
    smiles: Optional[str] = None
    unique_id: Optional[str] = None
    charge: int = _DEFAULT_CHARGE
    multiplicity: int = _DEFAULT_MULTIPLICITY
    property: Optional[dict] = None  # save molecule-level properties
    atom_property: Optional[list] = None  # save atom-level properties
    metadata: Optional[dict] = None  # save timings, optimization info, etc.

    def __post_init__(self):
        # Automatically generate a unique key if not provided after initialization
        if self.unique_id is None:
            self.unique_id = str(uuid4().int)[:_UNIQUE_KEY_LENGTH]
        if isinstance(self.xyz, list):
            self.xyz = np.array(self.xyz)
        if isinstance(self.elements, np.ndarray):
            self.elements = self.elements.tolist()

        if self.property is None:
            self.property = {}
        if self.atom_property is None:
            # Initialize atom_property and add atom symbol
            self.atom_property = [
                {'element': atom_symbol}
                for atom_symbol in self.elements] #yapf:disable
        if self.metadata is None:
            self.metadata = {}

    def save_gradients(self,
                       gradients_arr: np.ndarray,
                       prop_key: str = DFT_FORCES_KEY):
        """
        Save array that contains gradients for all atoms into per-atom dictionary
        :param gradients_arr: (n_atoms, 3) shape, gradients of each atom along x, y, z
        """
        for i, _ in enumerate(self.elements):
            self.atom_property[i][prop_key] = gradients_arr[i].tolist()

    def save_charges(self, charges_arr: np.ndarray, prop_key: str):
        """
        Save array that contains charges for all atoms into per-atom dictionary
        :param charges_arr: (n_atoms,) shape, one charge value for each atom
        :param prop_key: key to save the charge value; charges computed with different
                         methods can be saved with different keys
        """
        for i, _ in enumerate(self.elements):
            self.atom_property[i][prop_key] = float(charges_arr[i])

    def from_ase_atoms(cls, ase_atoms: Atoms):
        return cls(elements=ase_atoms.get_chemical_symbols(),
                   xyz=ase_atoms.get_positions(),
                   atomic_numbers=ase_atoms.get_atomic_numbers().tolist())

    def to_ase_atoms(self) -> Atoms:
        return Atoms(symbols=self.elements, positions=self.xyz)

    def to_pyscf_mole(self, basis: str) -> 'pyscf.gto.mole.Mole':
        """
        Convert the Structure object to a PySCF Mole object.

        :param basis: The basis set to be used for the PySCF calculation.
        :return: a `pyscf.gto.mole.Mole` object representing the structure.
        """

        atom_str = '\n'.join(f'{el} {x} {y} {z}'
                             for el, (x, y, z) in zip(self.elements, self.xyz))
        mol = pyscf.M(
            atom=atom_str,
            basis=basis,
            charge=self.charge,
            spin=self.multiplicity - 1,
            # Note that: mol.spin = 2S = Nalpha - Nbeta = unpaired_electrons (multiplicity=2S+1)
            unit=COORDINATE_UNIT)
        return mol

    def to_rdkit_mol(self, remove_hydrogens=False) -> 'Chem.rdchem.Mol':
        """
        Convert the Structure object to an RDKit Mol object.

        :param remove_hydrogens: Whether to remove hydrogens from the molecule.
                                 The canonical SMILES string will not have H atoms
        :return: an `rdkit.Chem.Mol` object representing the structure.
        """
        # Create an XYZ block from elements and xyz coordinates
        xyz_block = "\n".join(
            f"{el} {x:.f} {y:.f} {z:.f}"
            for el, (x, y, z) in zip(self.elements, self.xyz)) # yapf:disable

        # Create a molecule from the XYZ block
        mol = Chem.MolFromXYZBlock(xyz_block, sanitize=False)
        if mol is None:
            raise ValueError("Failed to create RDKit Mol from XYZ block.")

        # Resolve bonding info based on coordinates
        Chem.rdDetermineBonds.DetermineConnectivity(mol)

        if remove_hydrogens:
            return Chem.RemoveHs(mol)

        return mol

    def __eq__(self, other):
        if not isinstance(other, Structure):
            return NotImplemented

        elements_equal = self.elements == other.elements
        smiles_equal = self.smiles == other.smiles
        charge_equal = np.allclose(self.charge, other.charge)
        multiplicity_equal = np.allclose(self.multiplicity, other.multiplicity)

        # Check if properties are equal
        if self.property is not None and other.property is not None:
            for key in self.property:
                if key not in other.property:
                    return False
                if isinstance(self.property[key], np.ndarray):
                    if not np.allclose(self.property[key],
                                       other.property[key]):
                        return False
                else:
                    if self.property[key] != other.property[key]:
                        return False

        # unique_id is not compared as the structural equality (not identity) is desired
        # Special handling for numpy array comparison
        xyz_equal = np.allclose(
            self.xyz,
            other.xyz) if self.xyz.shape == other.xyz.shape else False

        return (elements_equal and xyz_equal and smiles_equal and charge_equal
                and multiplicity_equal)
