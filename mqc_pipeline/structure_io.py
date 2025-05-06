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

SUPPORTED_FORMATS = ['json', 'pickle', 'hdf5', 'xyz']


def write(structures: StructureType,
          file_or_path: FileLikeType,
          format: str = 'json'):
    """
    Write one or multiple Structure object(s) to a file in the specified format.

    :param structures: A single Structure object or an iterable of Structure objects
    :param file_or_path: Path to output file or file-like object
    :param format: Format to serialize to ('json', 'pickle', 'hdf5', 'xyz')

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
    if format == 'hdf5':
        write_hdf5(structures, file_or_path)
    if format == 'xyz':
        write_xyz(structures, file_or_path)


def read(file_or_path: FileLikeType, format: str = 'json') -> StructureType:
    """
    Read one or multiple Structure object(s) from a file in the given format.

    :param file_or_path: Path to input file or file-like object
    :param format: Format to deserialize from ('json', 'pickle', 'hdf5', 'xyz')
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
    if format == 'hdf5':
        return read_hdf5(file_or_path)
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


def write_hdf5(structures: StructureType, file_or_path):
    with h5py.File(file_or_path, 'w') as h5fhandle:
        if isinstance(structures, Structure):
            # Single structure
            _write_one_st_to_h5(structures, h5fhandle)
        else:
            # Collection of structures
            for st in structures:
                group = h5fhandle.create_group(f"structure_{st.unique_id}")
                _write_one_st_to_h5(st, group)


def _write_one_st_to_h5(structure: Structure, h5group: h5py.Group):
    """
    Helper method to write a single structure to an HDF5 group.
    """
    h5group.create_dataset('elements',
                           data=np.array(structure.elements, dtype='S10'))
    h5group.create_dataset('xyz', data=structure.xyz)

    if structure.atomic_numbers:
        h5group.create_dataset('atomic_numbers', data=structure.atomic_numbers)

    # Save attributes
    h5group.attrs['smiles'] = structure.smiles or ''
    h5group.attrs['unique_id'] = structure.unique_id
    h5group.attrs['charge'] = structure.charge
    h5group.attrs['multiplicity'] = structure.multiplicity

    # Save properties as a group
    if structure.property:
        prop_group = h5group.create_group('property')
        for key, value in structure.property.items():
            prop_group.create_dataset(key, data=value)


def read_hdf5(file_or_path) -> Structure:
    with h5py.File(file_or_path, 'r') as h5fhandle:
        # Check if it's a list or single structure
        st_group_keys = [
            k for k in h5fhandle.keys() if k.startswith('structure_')
        ]
        if len(st_group_keys) > 0:
            # multple structures
            structures = []
            for key in st_group_keys:
                group = h5fhandle[key]
                structures.append(_read_one_st_from_h5(group))
            return structures
        else:
            # for single structure
            return _read_one_st_from_h5(h5fhandle)


def _read_one_st_from_h5(h5group: h5py.Group) -> Structure:
    """
    Helper method to read a single structure from an HDF5 group.
    """
    elements = h5group['elements'][:].astype(str).tolist()
    xyz = h5group['xyz'][:]
    atomic_numbers = h5group['atomic_numbers'][:].tolist(
    ) if 'atomic_numbers' in h5group else None

    # Load attributes
    attrs = dict(h5group.attrs)
    smiles = attrs.get('smiles', None)
    unique_id = attrs['unique_id']
    charge = float(attrs['charge'])
    multiplicity = float(attrs['multiplicity'])

    # Load properties
    property_dict = {}
    if 'property' in h5group:
        prop_group = h5group['property']
        for key, value in prop_group.items():
            # Note: `shape == ()` for Scalar dataset
            property_dict[key] = value[()] if value.shape == (
            ) else value[:].tolist()

    return Structure(elements, xyz, atomic_numbers, smiles, unique_id, charge,
                     multiplicity, property_dict)


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
