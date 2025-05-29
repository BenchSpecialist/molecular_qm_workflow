"""
This module provides adaptors to streamline conversions between three main molecular
representations used in this package:
`Structure`, `rdkit.Chem.Mol`, and `pyscf.gto.Mole` objects.
making it easier to integrate tools like RDKit and PySCF into a unified workflow.

Usage:
------
from .adaptors import get_adaptor
from .common import Structure
from rdkit import Chem
from pyscf import M

# StructureAdaptor
st = Structure(elements=["H", "O", "H"], xyz=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
st_adaptor = get_adaptor(st)
rdkit_mol = st_adaptor.to_rdkit_mol()
pyscf_mole = st_adaptor.to_pyscf_mole(basis="sto-3g")

# RDKitMolAdaptor
rdkit_mol = Chem.MolFromSmiles('[OH-]')
adaptor = get_adaptor(rdkit_mol)
st = adaptor.to_structure()
pyscf_mole = adaptor.to_pyscf_mole(basis="sto-3g")

# PySCFMoleAdaptor
pyscf_mole = M(atom='He', basis='sto-3g')
adaptor = get_adaptor(pyscf_mole)
st = adaptor.to_structure()
rdkit_mol = adaptor.to_rdkit_mol()
"""

import numpy as np

from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from pyscf import gto, M

from .common import Structure, COORDINATE_UNIT, get_unpaired_electrons


class MoleculeAdaptor(ABC):

    @abstractmethod
    def to_structure(self):
        """Convert to a Structure object."""
        pass

    @abstractmethod
    def to_rdkit_mol(self, remove_hydrogens=False):
        """Convert to an RDKit Mol object."""
        pass

    @abstractmethod
    def to_pyscf_mole(self, basis: str):
        """Convert to a PySCF Mole object."""
        pass


class StructureAdaptor(MoleculeAdaptor):

    def __init__(self, st: Structure):
        self.st = st

    def to_structure(self):
        return self.st

    def to_rdkit_mol(self, remove_hydrogens=False):
        """
        Convert the Structure object to an RDKit Mol object.
        Note that the total formal charge is lost in this conversion, as RDKit
        doesn't allow setting the total formal charge for the whole molecule in
        a global way unless per-atom partial charges are set. So:
        `Chem.GetFormalCharge(converted_mol) != self.charge`

        :param remove_hydrogens: Whether to remove hydrogens from the molecule.
                                 The canonical SMILES string will not have H atoms
        :return: an `rdkit.Chem.rdchem.Mol` object representing the structure.
        """
        # Create an XYZ block from elements and xyz coordinates
        xyz_block = f"{len(self.st.elements)}\n\n" + "\n".join(
            f"{el} {x} {y} {z}"
            for el, (x, y, z) in zip(self.st.elements, self.st.xyz)) # yapf:disable
        # Create a molecule from the XYZ block
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            raise RuntimeError(
                "StructureAdaptor: Failed to create RDKit Mol from XYZ block.")

        # Resolve bonding info based on coordinates
        rdDetermineBonds.DetermineConnectivity(mol, charge=self.st.charge)

        if remove_hydrogens:
            return Chem.RemoveHs(mol)

        return mol

    def to_pyscf_mole(self, basis: str):
        return self.st.to_pyscf_mole(basis)


class RDKitMolAdaptor(MoleculeAdaptor):

    def __init__(self, rdkit_mol: Chem.Mol):
        if self._has_hydrogen(rdkit_mol):
            self.rdkit_mol = rdkit_mol
            self.mol_no_H = None
        else:
            # Add hydrogens as other types require all atoms
            self.rdkit_mol = Chem.AddHs(rdkit_mol)
            # Save the original molecule without hydrogens for the mapping to itself
            self.mol_no_H = rdkit_mol

    @staticmethod
    def _has_hydrogen(rdkit_mol):
        return any(atom.GetAtomicNum() == 1 for atom in rdkit_mol.GetAtoms())

    def to_structure(self):
        xyz_block = Chem.MolToXYZBlock(self.rdkit_mol)
        # Empty string when the input conformer has no conformer
        if not xyz_block:
            raise RuntimeError(
                "RDKitMolAdaptor: Input RDKit molecule has no conformer and cannot be written to XYZ block."
            )
        atom_coords = xyz_block.split("\n\n")[-1].splitlines()
        elements, xyz = [], []
        for line in atom_coords:
            parts = line.split()
            # ignores additional columns beyond the first 4
            elements.append(parts[0])
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return Structure(elements=elements,
                         xyz=np.array(xyz),
                         charge=Chem.GetFormalCharge(self.rdkit_mol),
                         multiplicity=1 +
                         get_unpaired_electrons(self.rdkit_mol))

    def to_rdkit_mol(self, remove_hydrogens=False):
        if self.rdkit_mol.GetNumBonds() == 0:
            rdDetermineBonds.DetermineConnectivity(self.rdkit_mol)

        if remove_hydrogens:
            if self.mol_no_H is None:
                return Chem.RemoveHs(self.rdkit_mol)
            else:
                return self.mol_no_H
        return self.rdkit_mol

    def to_pyscf_mole(self, basis: str) -> gto.Mole:
        """
        Convert an RDKit Mol object (contains hydrogen) to a PySCF Mole object.

        :param mol: An RDKit Mol object.
        :param basis: The basis set to be used for the PySCF calculation.

        :return: a `pyscf.gto.mole.Mole` object representing the structure.
        """
        atom_str = Chem.MolToXYZBlock(self.rdkit_mol).split("\n\n")[-1]
        # Empty string when the input conformer has no conformer
        if not atom_str:
            raise RuntimeError(
                "RDKitMolAdaptor: Input RDKit molecule has no conformer and cannot be written to XYZ block."
            )

        pyscf_mole = M(atom=atom_str,
                       basis=basis,
                       charge=Chem.GetFormalCharge(self.rdkit_mol),
                       spin=get_unpaired_electrons(self.rdkit_mol),
                       unit=COORDINATE_UNIT)
        return pyscf_mole


class PySCFMoleAdaptor(MoleculeAdaptor):

    def __init__(self, pyscf_mole: gto.Mole):
        self.pyscf_mole = pyscf_mole

    def to_structure(self):
        mol = self.pyscf_mole
        xyz = mol.atom_coords(unit=COORDINATE_UNIT)
        elements = [mol.atom_symbol(i) for i in range(mol.natm)]
        return Structure(
            elements=elements,
            xyz=xyz,
            atomic_numbers=[mol.atom_charge(i) for i in range(mol.natm)],
            charge=mol.charge,
            multiplicity=mol.multiplicity)

    def to_rdkit_mol(self, remove_hydrogens=False):
        mol_xyz = self.pyscf_mole.tostring(format="xyz")
        rdkit_mol = Chem.MolFromXYZBlock(mol_xyz)

        if rdkit_mol is None:
            raise RuntimeError(
                "PySCFMoleAdaptor: Failed to convert PySCF molecule to RDKit molecule."
            )

        rdDetermineBonds.DetermineBonds(rdkit_mol,
                                        charge=self.pyscf_mole.charge)
        if remove_hydrogens:
            return Chem.RemoveHs(rdkit_mol)
        return rdkit_mol

    def to_pyscf_mole(self, basis: str):
        return self.pyscf_mole


def get_adaptor(obj):
    if isinstance(obj, Structure):
        return StructureAdaptor(obj)
    elif isinstance(obj, Chem.Mol):
        return RDKitMolAdaptor(obj)
    elif isinstance(obj, gto.Mole):
        return PySCFMoleAdaptor(obj)
    else:
        raise TypeError("Unsupported molecule type.")
