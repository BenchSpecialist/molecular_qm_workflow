import time
import numpy as np
from pathlib import Path
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdMolTransforms import GetBondLength

from .constants import chemical_symbols
from .common import Structure, get_unpaired_electrons
from .adaptors import get_adaptor
from .util import get_default_logger

logger = get_default_logger()

_RDKIT_MAX_ATTEMPTS = 100
_PYBEL_FORCEFILED = "mmff94"


def smiles_to_structure(smiles: str,
                        method: str = 'openbabel',
                        **kwargs) -> Structure:
    """
    Convert a SMILES string to a 3D structure using the specified method.

    :param smiles: A SMILES string representing the molecule.
    :param method: Method to use for structure generation, either 'rdkit' or 'openbabel'.

    :return: A Structure object containing the elements, 3D coordinates, and SMILES string.
    """
    if method == "rdkit":
        return smiles_to_structure_rdk(smiles,
                                       max_attempts=kwargs.get(
                                           "max_attempts",
                                           _RDKIT_MAX_ATTEMPTS))
    elif method == "openbabel":
        return smiles_to_structure_pybel(smiles,
                                         forcefield=kwargs.get(
                                             "forcefield", _PYBEL_FORCEFILED))
    else:
        raise ValueError(f'smiles_to_structure: "{method}" is not supported.')


@lru_cache(maxsize=1)
def _import_pybel():
    """Import pybel module once and cache it."""
    from openbabel import pybel
    return pybel


def smiles_to_structure_pybel(smiles: str,
                              forcefield=_PYBEL_FORCEFILED) -> Structure:
    """
    Convert a SMILES string to a 3D structure using OpenBabel.

    :param smiles: A SMILES string representing the molecule.
    :return: A Structure object containing the elements, 3D coordinates, and SMILES string.
    """
    try:
        pybel = _import_pybel()
    except ImportError as e:
        err_msg = "Function smiles_to_structure_pybel needs openbabel package, which is not installed."
        logger.error(err_msg)
        raise ImportError(err_msg) from e

    t_start = time.perf_counter()
    try:
        mol = pybel.readstring("smi", smiles)
    except Exception as e:
        err_msg = f"smiles_to_structure_pybel: {str(e)}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Generate 3D coordinates: hydrogens are added and a quick local optimization
    # is carried out with 50 steps using MMFF94 forcefield.
    mol.make3D(forcefield=_PYBEL_FORCEFILED, steps=50)

    # Improve the coordinates further
    mol.localopt(forcefield=forcefield, steps=500)

    # Check if the OpenBabel-generated geometry is physical:
    # bond distances should be in the range of 0.5 angstrom ~ 3 angstrom
    ob_mol = mol.OBMol
    bond_distances = [
        np.linalg.norm(
            np.array(mol.atoms[bond.GetBeginAtomIdx() - 1].coords) -
            np.array(mol.atoms[bond.GetEndAtomIdx() - 1].coords))
        for bond in pybel.ob.OBMolBondIter(ob_mol)
    ]
    if any(distance < 0.5 or distance > 3.0 for distance in bond_distances):
        bd_err_msg = f"{smiles}: Unphysical bond distances in the OpenBabel conformer.\n{bond_distances}"
        logger.error(bd_err_msg)
        raise RuntimeError(bd_err_msg)

    metadata = {
        "openbabel_time": round(time.perf_counter() - t_start, 4),
    }

    return Structure(
        elements=[chemical_symbols[atom.atomicnum] for atom in mol.atoms],
        xyz=np.array([atom.coords for atom in mol.atoms]),
        smiles=smiles,
        charge=mol.charge,
        multiplicity=mol.spin,
        metadata=metadata)


def smiles_to_structure_rdk(smiles: str,
                            max_attempts: int = _RDKIT_MAX_ATTEMPTS
                            ) -> Structure:
    """
    Convert a SMILES string to a 3D structure using RDKit.

    :param smiles: A SMILES string representing the molecule.
    :param max_attempts: Maximum attempts for embedding with RDKit.

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
        "rdkit_time": round(time.perf_counter() - t_start, 4),
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
        logger.warning(
            'get_canonical_smiles_rdk: Using `Chem.MolFromXYZFile` with `rdDetermineBonds.DetermineConnectivity` '
            'may not assign overall charge correctly.')
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

    :param xyz_file_or_block: Path to XYZ file or XYZ format string
    :return: Canonical SMILES string
    :raises ImportError: If openbabel package is not installed
    """
    try:
        pybel = _import_pybel()
    except ImportError as e:
        err_msg = "Function get_canonical_smiles_ob needs openbabel package, which is not installed."
        logger.error(err_msg)
        raise ImportError(err_msg) from e

    if xyz_file_or_block.endswith(".xyz"):
        obmol = next(pybel.readfile("xyz", xyz_file_or_block))
    else:
        # Convert XYZ block to Open Babel molecule
        obmol = pybel.readstring("xyz", xyz_file_or_block)

    obmol.addh()  # add hydrogens
    obmol.make3D()  # refine 3D geometry
    return obmol.write("can").strip().split('\t')[0]


def get_canonical_smiles_for_xyz(xyz_block: str) -> str:
    """
    Get the RDKit-defined canonical SMILES string for a 3D structure represented
    by an XYZ block.

    :param xyz_block: A string containing the XYZ format of the molecule.
    :return: Canonical SMILES string in the RDKit definition.
    """
    ob_can_smiles = get_canonical_smiles_ob(xyz_block)
    rdkit_mol = Chem.MolFromSmiles(ob_can_smiles)
    return Chem.MolToSmiles(rdkit_mol, isomericSmiles=False, canonical=True)


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


def generate_optimized_rdk_confs(
    input: Chem.rdchem.Mol | str,
    target_n_conformers: int = 20,
    rmsd_threshold: float = 0.5,
    max_attempts: int = 1000
) -> tuple[Chem.rdchem.Mol, list[tuple[int, float]]]:
    """
    Generate multiple optimized conformers for a given RDKit molecule or SMILES string.
    The function uses RDKit to generate conformers and optimize them using MMFF.
    :param input: A SMILES string or an RDKit Mol object.
    :param target_n_conformers: Target number of conformers to generate, upper
                                bound of the final number of conformers returned.
    :param rmsd_threshold: RMSD threshold for pruning conformers.
    :param max_attempts: Maximum number of attempts to generate conformers.
    :return: A tuple containing the RDKit Mol object with conformers and a list of
             sorted conformer ID tuple (ID, final energy in kcal/mol) in ascending
             order of their final energies.
    """
    # Check if the input is a valid SMILES string or an RDKit Mol object
    if isinstance(input, str):
        smiles = input.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        mol = Chem.AddHs(mol)
    elif isinstance(input, Chem.rdchem.Mol):
        mol = input
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    else:
        raise TypeError(
            "generate_optimized_rdk_confs: Input must be a SMILES string or an RDKit Mol object."
        )

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = rmsd_threshold
    params.randomSeed = 42
    params.numThreads = 0  # Use all available CPUs
    params.useBasicKnowledge = True
    params.useExpTorsionAnglePrefs = True
    params.enforceChirality = False
    params.useSmallRingTorsions = False
    try:
        params.useRotatableBondPruning = True
        params.maxAttempts = max_attempts
    except AttributeError:
        pass

    # Generate multiple conformers
    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol,
                                              numConfs=target_n_conformers,
                                              params=params)
    except Exception as e:
        logger.warning(
            f"AllChem.EmbedMultipleConfs failed to generate conformers for {smiles}: {str(e)}"
        )
        return mol, []

    if not conf_ids:
        logger.warning(
            f"generate_optimized_rdk_confs: No conformers generated for {smiles}"
        )
        return mol, []

    # Optimize each conformer
    conf_id_energies = []
    for conf_id in conf_ids:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is not None:
            ff.Minimize()
            final_energy = ff.CalcEnergy()
            conf_id_energies.append((conf_id, final_energy))

    # Sort conformer index by energy in ascending order
    conf_id_energies.sort(key=lambda x: x[1])
    return mol, conf_id_energies
