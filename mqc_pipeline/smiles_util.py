import time
import logging
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
    t_start = time.perf_counter()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add H atoms to the molecule
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    # With fixed seed, the generated coordinates are reproducible, assuming:
    # 1. The same version of RDKit; 2. Same input SMILES (with consistent atom ordering)
    # 3. No other sources of nondeterminism, like inconsistent hydrogen handling
    # The value of randomSeed can affect the success rate of EmbedMolecule, which
    # uses a distance geometry algorithm that samples random initial coordinates
    # guided by distance bounds, where the random number generator is used.
    params.randomSeed = 42

    # Embed the molecule
    for attempt in range(max_attempts):
        if AllChem.EmbedMolecule(mol, params) == 0:
            logging.info(f"Embedding succeeded: {smiles}")
            break

        # Within the same attempt, try disabling chirality if embedding fails
        # due to chirality constraints.
        params.enforceChirality = False
        if AllChem.EmbedMolecule(mol, params) == 0:
            logging.info(
                f"Embedding succeeded after disabling chirality: {smiles}")
            # Reset the chirality flag for the next attempt
            params.enforceChirality = True
            break

        if attempt == max_attempts - 1:
            raise ValueError(
                f"Failed to embed after {max_attempts} attempts: {smiles}")

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

    # Compute the number of unpaired electrons
    unpaired_electrons = sum(atom.GetNumRadicalElectrons()
                             for atom in mol.GetAtoms())
    # Compute the charge
    charge = Chem.GetFormalCharge(mol)

    metadata = {
        "rdkit_duration": time.perf_counter() - t_start,
        "rdkit_attempts": attempt + 1,
    }

    return Structure(elements=elements,
                     xyz=xyz,
                     smiles=smiles,
                     charge=charge,
                     multiplicity=unpaired_electrons + 1,
                     metadata=metadata)
