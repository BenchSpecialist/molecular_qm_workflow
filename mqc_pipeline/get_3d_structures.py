import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength

from .common import Structure

MAX_ATTEMPTS_EMBED = 100


def smiles_to_3d_structures_by_rdkit(smiles: str,
                                     max_attempts: float = MAX_ATTEMPTS_EMBED
                                     ) -> Structure:
    """
    Convert a SMILES string to a 3D structure using RDKit.

    :param smiles: A SMILES string representing the molecule.
    :return: A Structure object containing the elements, 3D coordinates, and SMILES string.

    :raises ValueError: If the SMILES string is invalid or if embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add H atoms to the molecule
    mol = Chem.AddHs(mol)

    # Embed the molecule
    for attempt in range(max_attempts):
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0:
            break
        # If we failed once, try disabling chirality
        ps = AllChem.ETKDGv3()
        ps.enforceChirality = False
        if AllChem.EmbedMolecule(mol, ps) == 0:
            print(f"Success after disabling chirality for {smiles}")
            ps.enforceChirality = True  # Re-enable to let the next molecule try chirality
            break
        if attempt == max_attempts - 1:
            raise ValueError(
                f"Failed to embed molecule for SMILES after {max_attempts} attempts: {smiles}"
            )

    # Get conformer with 3D coordinates
    conf = mol.GetConformer()

    # Check if the RDKit-generated geometry is physical:
    # bond distances should be in the range of 0.5 angstrom ~ 3 angstrom
    bond_distances = [
        GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ]
    if any(distance < 0.5 or distance > 3.0 for distance in bond_distances):
        raise ValueError(
            f"Unphysical bond distances in the RDKit-generated 3D structure for {smiles}."
        )

    # Extract coordinates
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz[i] = [pos.x, pos.y, pos.z]

    return Structure(elements=elements, xyz=xyz, smiles=smiles)
