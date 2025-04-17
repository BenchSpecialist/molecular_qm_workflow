from dataclasses import dataclass
import numpy as np
from ase import Atoms 

@dataclass
class Molecule3DInfo:
    elements: list[str]
    xyz: np.ndarray    # shape (n_atoms, 3)
    smiles: str = None
    key: str = None

    def from_ase_atoms(cls, atoms: Atoms):
        return cls(elements=atoms.get_chemical_symbols(),
                   xyz=atoms.get_positions())
    
    def to_ase_atoms(self) -> Atoms:
        return Atoms(
            symbols=self.elements, 
            positions=self.xyz
        )
    
    