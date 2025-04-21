import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms
from uuid import uuid4

_DEFAULT_CHARGE = 0
_DEFAULT_MULTIPLICITY = 1
_UNIQUE_KEY_LENGTH = 11


@dataclass
class Structure:
    """
    Container to hold 3D information of a molecule.
    """
    elements: list[str]
    xyz: np.ndarray  # shape (n_atoms, 3)
    atomic_numbers: Optional[list[int]] = None
    smiles: Optional[str] = None
    unique_id: Optional[str] = None
    charge: int = _DEFAULT_CHARGE
    multiplicity: int = _DEFAULT_MULTIPLICITY
    property: Optional[dict] = None

    def __post_init__(self):
        # automatically generate a unique key if not provided after initialization
        if self.unique_id is None:
            self.unique_id = str(uuid4().int)[:_UNIQUE_KEY_LENGTH]
        if isinstance(self.xyz, list):
            self.xyz = np.array(self.xyz)

    def from_ase_atoms(cls, ase_atoms: Atoms):
        return cls(elements=ase_atoms.get_chemical_symbols(),
                   xyz=ase_atoms.get_positions(),
                   atomic_numbers=ase_atoms.get_atomic_numbers().tolist())

    def to_ase_atoms(self) -> Atoms:
        return Atoms(symbols=self.elements, positions=self.xyz)

    def write_xyz(self, xyz_file_path: str):
        """
        Save cartesian coordinates of a molecule to an XYZ file. If any energy
        information is available, it will be included in the comment line of the
        XYZ file.

        :param xyz_file_path: Path to the output XYZ file.
        """
        energy_keys = [k for k in self.property if "energy" in k]
        energy_info = ', '.join(
            [f"{k}: {self.property[k]:.6f} Eh"
             for k in energy_keys]) if energy_keys else ""

        with open(xyz_file_path, 'w') as xyz_file:
            xyz_file.write(f"{len(self.elements)}\n")
            xyz_file.write(
                f"Optimized geometry for molecule {self.unique_id}, {energy_info}\n"
            )
            for element, coord in zip(self.elements, self.xyz):
                xyz_file.write(
                    f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
                )

    def __eq__(self, other):
        if not isinstance(other, Structure):
            return NotImplemented

        elements_equal = self.elements == other.elements
        smiles_equal = self.smiles == other.smiles
        charge_equal = self.charge == other.charge
        multiplicity_equal = self.multiplicity == other.multiplicity
        property_equal = self.property == other.property
        # unique_id is not compared as the structural equality (not identity) is desired
        # Special handling for numpy array comparison
        xyz_equal = np.allclose(
            self.xyz, other.xyz, rtol=1e-5,
            atol=1e-8) if self.xyz.shape == other.xyz.shape else False

        return (elements_equal and xyz_equal and property_equal
                and smiles_equal and charge_equal and multiplicity_equal)
