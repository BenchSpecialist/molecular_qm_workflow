"""
This module contains:
- Structure class: A container for molecular structures, including methods for
  conversion to and from different formats (ASE, RDKit, PySCF). It's the central
  data structure for the pipeline, which save calculated geometries, properties
  and metadata.
- Utility functions:
    - calculation of unpaired electrons from a RDKit molecule.
"""

import numpy as np
from uuid import uuid4
from typing import Optional
from dataclasses import dataclass

from ase import Atoms
import pyscf

COORDINATE_UNIT = 'angstrom'
_DEFAULT_CHARGE = 0
_DEFAULT_MULTIPLICITY = 1
_UNIQUE_KEY_LENGTH = 16

from .constants import HARTREE_TO_EV, ELEMENT_TO_ATOMIC_NUMBER


@dataclass(slots=True)  # eliminates the __dict__ to reduce memory overhead
class Structure:
    """
    Container to hold information of a 3D molecule.
    """
    elements: list[str]
    xyz: np.ndarray  # shape (n_atoms, 3)
    atomic_numbers: Optional[list[int]] = None
    smiles: str = ""
    unique_id: Optional[str] = None
    charge: int = _DEFAULT_CHARGE
    multiplicity: int = _DEFAULT_MULTIPLICITY
    property: Optional[dict] = None  # save molecule-level properties
    atom_property: Optional[dict] = None  # save atom-level properties
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
            # Values in atom_property dict should be lists/1D arrays, in which
            # the element order is same as the order of atom symbols in the
            # `self.elements` list.
            self.atom_property = {}
        if self.metadata is None:
            self.metadata = {}

    def save_gradients(self, gradients_arr: np.ndarray, prop_key: str):
        """
        Save gradients per-axis into the atom_property dictionary.
        """
        self.atom_property[f'{prop_key}_x'] = gradients_arr[:, 0].tolist()
        self.atom_property[f'{prop_key}_y'] = gradients_arr[:, 1].tolist()
        self.atom_property[f'{prop_key}_z'] = gradients_arr[:, 2].tolist()

    def save_thermo_info(self, thermo_info: dict) -> None:
        """
        Save essential thermochemical properties to the structure properties.
        """
        _e_keys = ('ZPE', 'E_0K', 'E_tot', 'H_tot', 'G_tot')
        self.property.update({
            f'{key.lower()}_eV':
            float(value[0] * HARTREE_TO_EV)
            for key in _e_keys if (value := thermo_info.get(key)) is not None
        })

        if (cv_data := thermo_info.get('Cv_tot')):
            cv_value, unit = cv_data
            self.property[f'Cv_tot_{unit}'] = float(cv_value)

    @classmethod
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

    def to_xyz_block(self) -> str:
        """
        Convert the Structure object to an XYZ formatted string.

        :return: A string in XYZ format representing the structure.
        """
        comment_line = f"{self.smiles} unique_id:{self.unique_id} charge:{self.charge} multiplicity:{self.multiplicity}\n"

        xyz_lines = [f"{len(self.elements)}\n", comment_line]
        for el, (x, y, z) in zip(self.elements, self.xyz):
            xyz_lines.append(f"{el} {x} {y} {z}\n")
        return ''.join(xyz_lines)

    @classmethod
    def from_xyz_block(
            cls,
            xyz_block: str,
            partial_charge_column: int | None = None) -> 'Structure':
        """
        Create a Structure object from an XYZ formatted string.

        :param xyz_block: A string in XYZ format.
        :param partial_charge_column: The column index (0-based) where the partial
                                      charge is located in an extended XYZ format.
        :return: A Structure object initialized with the data from the XYZ block.
        """
        lines = xyz_block.strip().split('\n')
        elements, xyz, partial_charges = [], [], []
        for i, line in enumerate(lines[2:]):
            parts = line.split()
            elements.append(parts[0])
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if partial_charge_column is None:
                continue  # Skip charge extraction if not specified

            # Extract partial charge from the specified column
            if len(parts) > partial_charge_column:
                charge = float(parts[partial_charge_column])
                partial_charges.append(charge)
            else:
                raise ValueError(
                    f"Line {i+1} doesn't have enough columns for charge data")

        if partial_charges:
            # Calculate total molecular charge from partial charges
            total_charge = round(np.sum(partial_charges))
            return cls(elements=elements,
                       xyz=np.array(xyz),
                       charge=total_charge)

        return cls(elements=elements, xyz=np.array(xyz))

    @staticmethod
    def get_unpaired_electrons(atom_numbers_or_eles: list[int] | list[str],
                               charge: int) -> int:
        """
        Calculate the number of unpaired electrons in a molecule based on
        atomic numbers or element symbols.

        :param atom_numbers_or_eles: List of atomic numbers or element symbols.
        :param charge: Total charge of the molecule
        :return: Number of unpaired electrons (0 or 1)
        """
        if isinstance(atom_numbers_or_eles[0], str):
            # Convert element symbols to atomic numbers
            atom_numbers = [
                ELEMENT_TO_ATOMIC_NUMBER[el] for el in atom_numbers_or_eles
            ]
        else:
            atom_numbers = atom_numbers_or_eles
        # Only the parity matters for unpaired electrons calculation
        # So we can take modulo 2 before summing to avoid overflow
        return (sum(num % 2 for num in atom_numbers) - charge % 2) % 2

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


def get_unpaired_electrons(rdk_mol) -> int:
    """
    Compute the number of unpaired electrons in a RDKit molecule.

    :param mol: `rdkit.Chem.rdchem.Mol` object representing the molecule.

    :return: The number of unpaired electrons in the molecule.
    """
    return sum(atom.GetNumRadicalElectrons() for atom in rdk_mol.GetAtoms())
