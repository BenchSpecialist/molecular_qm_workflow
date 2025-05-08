"""
Module for writing and reading Structure object(s) to/from various file formats.
"""

import json
import pickle
import h5py
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Union, BinaryIO, TextIO, Iterable

from .common import Structure

# Type aliases
StructureType = Union[Structure, Iterable[Structure]]
FileLikeType = Union[str, Path, TextIO, BinaryIO]

SUPPORTED_FORMATS = ['json', 'pickle', 'xyz']


def write(structures: StructureType,
          file_or_path: FileLikeType,
          format: str = 'json'):
    """
    Write one or multiple Structure object(s) to a file in the specified format.

    :param structures: A single Structure object or an iterable of Structure objects
    :param file_or_path: Path to output file or file-like object
    :param format: Format to serialize to ('json', 'pickle', 'xyz')

    :raises ValueError: If the format is not supported
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats are: "
            f"{SUPPORTED_FORMATS}")
    if format == 'json':
        write_json(structures, file_or_path)
    if format == 'pickle':
        write_pickle(structures, file_or_path)
    if format == 'xyz':
        write_xyz(structures, file_or_path)


def read(file_or_path: FileLikeType, format: str = 'json') -> StructureType:
    """
    Read one or multiple Structure object(s) from a file in the given format.

    :param file_or_path: Path to input file or file-like object
    :param format: Format to deserialize from ('json', 'pickle', 'xyz')
    """
    format = format.lower()
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats are: "
            f"{SUPPORTED_FORMATS}")
    if format == 'json':
        return read_json(file_or_path)
    if format == 'pickle':
        return read_pickle(file_or_path)
    if format == 'xyz':
        return read_xyz(file_or_path)


def write_json(structures: StructureType, file_or_path: FileLikeType):
    if isinstance(structures, Structure):
        # Single structure
        data = asdict(structures)
        data['xyz'] = structures.xyz.tolist()
    else:
        # Collection of structures
        data = []
        for structure in structures:
            struct_data = asdict(structure)
            struct_data['xyz'] = structure.xyz.tolist()
            data.append(struct_data)

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        json.dump(data, file_or_path, indent=2)


def read_json(file_or_path: FileLikeType) -> StructureType:
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'r') as fhandle:
            data = json.load(fhandle)
    else:
        data = json.load(file_or_path)

    if isinstance(data, list):
        # Collection of structures
        structures = []
        for struct_data in data:
            struct_data['xyz'] = np.array(struct_data['xyz'])
            structures.append(Structure(**struct_data))
        return structures
    else:
        # Single structure
        data['xyz'] = np.array(data['xyz'])
        return Structure(**data)


def write_pickle(structure: StructureType, file_or_path: FileLikeType):
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'wb') as fhandle:
            pickle.dump(structure, fhandle)
    else:
        pickle.dump(structure, file_or_path)


def read_pickle(file_or_path: FileLikeType) -> StructureType:
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'rb') as fhandle:
            return pickle.load(fhandle)
    else:
        return pickle.load(file_or_path)


def write_xyz(structure: Structure, file_or_path):
    """
    Save cartesian coordinates of one Structure to an XYZ file. If any energy
    information is available, it will be included in the comment line of the
    XYZ file.
    """
    energy_keys = [k for k in structure.property
                   if "energy" in k] if structure.property else []
    energy_info = ', '.join(
        [f"{k}: {structure.property[k]:.6f} Eh"
         for k in energy_keys]) if energy_keys else ""

    with open(file_or_path, 'w') as xyz_file:
        xyz_file.write(f"{len(structure.elements)}\n")
        xyz_file.write(
            f"Optimized geometry for molecule {structure.unique_id} {energy_info}\n"
        )
        for element, coord in zip(structure.elements, structure.xyz):
            xyz_file.write(
                f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def read_xyz(file_or_path) -> Structure:
    """
    Read one Structure object from an XYZ file.
    """
    with open(file_or_path, 'r') as xyz_file:
        lines = xyz_file.readlines()

    # Extract info from comment line;
    # currently we assume the comment line contains: unique_id, energy
    comment = lines[1].strip().split()
    try:
        unique_id = comment[4]
        if comment[-1] == 'Eh':
            energy_val = comment[-2]
            energy_key = comment[-3].rstrip(':')
        property = {energy_key: energy_val}
    except:
        unique_id = None
        property = None

    elements = []
    xyz = []
    num_atoms = int(lines[0].strip())
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        elements.append(parts[0])
        xyz.append([float(coord) for coord in parts[1:]])

    xyz = np.array(xyz)
    return Structure(elements=elements,
                     xyz=xyz,
                     unique_id=unique_id,
                     property=property)
