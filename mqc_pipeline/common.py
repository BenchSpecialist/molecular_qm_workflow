from dataclasses import dataclass
import numpy as np
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
    smiles: str = None
    unique_id: str = None
    energy: float = None
    energy_method: str = None
    # TODO: save per-atom formal_charge
    charge: int = _DEFAULT_CHARGE
    multiplicity: int = _DEFAULT_MULTIPLICITY
    property: dict = None

    def __post_init__(self):
        # automatically generate a unique key if not provided after initialization
        if self.unique_id is None:
            self.unique_id = str(uuid4().int)[:_UNIQUE_KEY_LENGTH]

    def from_ase_atoms(cls, atoms: Atoms):
        return cls(elements=atoms.get_chemical_symbols(),
                   xyz=atoms.get_positions())

    def to_ase_atoms(self) -> Atoms:
        return Atoms(symbols=self.elements, positions=self.xyz)

    def write_xyz(self, file_path: str):
        """
        Save cartesian coordinates of a molecule to an XYZ file.
        
        :param file_path: Path to the output XYZ file.
        """
        with open(file_path, 'w') as xyz_file:
            xyz_file.write(f"{len(self.elements)}\n")
            xyz_file.write(
                f"Optimized geometry for molecule {self.unique_id}, energy: {self.energy:.6f} eV\n"
            )
            for element, coord in zip(self.elements, self.xyz):
                xyz_file.write(
                    f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
                )
