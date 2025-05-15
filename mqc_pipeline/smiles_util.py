import time
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdMolTransforms import GetBondLength
from openbabel import pybel

from .common import Structure, get_unpaired_electrons
from .adaptors import get_adaptor
from .util import logger

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
            break

        # Within the same attempt, try disabling chirality if embedding fails
        # due to chirality constraints.
        params.enforceChirality = False
        if AllChem.EmbedMolecule(mol, params) == 0:
            logger.info(
                f"{smiles}: Embedding succeeded after disabling chirality.")
            # Reset the chirality flag for the next attempt
            params.enforceChirality = True
            break

        if attempt == max_attempts - 1:
            err_msg = f"{smiles}: Embedding failed after {max_attempts} attempts."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    # Get conformer with 3D coordinates
    conf = mol.GetConformer()

    # Check if the RDKit-generated geometry is physical:
    # bond distances should be in the range of 0.5 angstrom ~ 3 angstrom
    bond_distances = [
        GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ]
    if any(distance < 0.5 or distance > 3.0 for distance in bond_distances):
        bd_err_msg = f"{smiles}: Unphysical bond distances in the RDKit conformer.\n{bond_distances}"
        logger.error(bd_err_msg)
        raise RuntimeError(bd_err_msg)

    # Extract coordinates
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz[i] = [pos.x, pos.y, pos.z]

    metadata = {
        "rdkit_duration": time.perf_counter() - t_start,
        "rdkit_attempts": attempt + 1,
    }

    return Structure(elements=elements,
                     xyz=xyz,
                     smiles=smiles,
                     charge=Chem.GetFormalCharge(mol),
                     multiplicity=1 + get_unpaired_electrons(mol),
                     metadata=metadata)


def get_canonical_smiles_rdk(input) -> str:
    """
    Convert a 3D structure to a canonical SMILES string using RDKit.
    Supported input types (contains coordinates) included:
        - XYZ file
        - Structure object
        - rdkit.Chem.Mol object
        - pyscf.gto.mole.Mole object
    """
    if isinstance(input, str) and Path(input).suffix == ".xyz":
        rdkit_mol = Chem.MolFromXYZFile(input)
        # Generate bonding information based on coordinates
        # xyz file doesn't have charge info
        rdDetermineBonds.DetermineConnectivity(rdkit_mol)
    else:
        try:
            # Get adaptor based on the input type
            adaptor = get_adaptor(input)
        except Exception:
            raise RuntimeError(f"Unsupported input type: {type(input)}. ")
        # Convert to RDKit Mol object, remove H atoms for canonical SMILES
        rdkit_mol = adaptor.to_rdkit_mol(remove_hydrogens=True)

    return Chem.CanonSmiles(Chem.MolToSmiles(rdkit_mol))


def get_canonical_smiles_ob(xyz_file_or_block) -> str:
    """
    Convert a 3D structure to a canonical SMILES string using Open Babel.
    """
    if xyz_file_or_block.endswith(".xyz"):
        obmol = next(pybel.readfile("xyz", xyz_file_or_block))
    else:
        # Convert XYZ block to Open Babel molecule
        obmol = pybel.readstring("xyz", xyz_file_or_block)

    obmol.addh()  # add hydrogens
    obmol.make3D()  # refine 3D geometry
    return obmol.write("can").strip().split('\t')[0]


def smiles_has_broken_bonds(smiles) -> bool:
    """
    Check if a SMILES string has broken bonds (i.e., disconnected fragments).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Get connected components (fragments)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return len(fragments) > 1
