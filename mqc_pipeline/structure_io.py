"""
Module for writing and reading Structure object(s) to/from various file formats.
"""
import re
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import asdict
from typing import Union, BinaryIO, TextIO, Iterable

from .common import Structure
from .property import DFT_ENERGY_KEY
from .constants import ELEMENT_TO_ATOMIC_NUMBER

# Type aliases
StructureType = Union[Structure, Iterable[Structure]]
FileLikeType = Union[str, Path, TextIO, BinaryIO]

# File type that save Structure object(s) to disk
SUPPORTED_OBJ_FILE_FORMATS = ['json', 'pickle', 'xyz']

# Columnar file format used to save Structure object(s) based on output schema
COLUMNAR_FILE_EXTENSIONS = ('.csv', '.parquet', '.parq')

SHARED_KEYS = ('unique_id', 'smiles')


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
    if format not in SUPPORTED_OBJ_FILE_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats are: "
            f"{SUPPORTED_OBJ_FILE_FORMATS}")
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
    if format not in SUPPORTED_OBJ_FILE_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats are: "
            f"{SUPPORTED_OBJ_FILE_FORMATS}")
    if format == 'json':
        return read_json(file_or_path)
    if format == 'pickle':
        return read_pickle(file_or_path)
    if format == 'xyz':
        return read_xyz(file_or_path)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(structures: StructureType, file_or_path: FileLikeType):
    if isinstance(structures, Structure):
        # Single structure
        data = asdict(structures)
    else:
        data = [asdict(st) for st in structures]

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    else:
        json.dump(data, file_or_path, indent=2, cls=NumpyEncoder)


def read_json(file_or_path: FileLikeType) -> StructureType:
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, 'r') as fhandle:
            data = json.load(fhandle)
    else:
        data = json.load(file_or_path)

    if isinstance(data, list):
        # Collection of structures
        sts = [Structure(**st_data) for st_data in data]
        return sts
    else:
        # Single structure
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


def write_xyz(st: Structure, xyz_path: str):
    """
    Save cartesian coordinates of one Structure to an XYZ file. If any energy
    information is available, it will be included in the comment line of the
    XYZ file.
    """
    energy_info = ""
    if dft_energy := st.property.get(DFT_ENERGY_KEY, None):
        energy_info = f"{DFT_ENERGY_KEY}:{dft_energy}"

    comment_line = f"{st.smiles} unique_id:{st.unique_id} charge:{st.charge} multiplicity:{st.multiplicity} {energy_info}\n"

    with open(xyz_path, 'w') as fhandle:
        fhandle.write(f"{len(st.elements)}\n")
        fhandle.write(comment_line)
        for element, (x, y, z) in zip(st.elements, st.xyz):
            fhandle.write(f"{element} {x} {y} {z}\n")


def read_xyz(xyz_path: str, parse_comment=False) -> Structure:
    """
    Read one Structure object from an XYZ file.

    :param xyz_path: Path to the XYZ file
    :param parse_comment: If True, parse the comment line for additional
                          information (e.g., unique_id, charge, multiplicity)
                          This is useful for getting additional info from xyz
                          files generated by `write_xyz` function in the same module.
    """
    with open(xyz_path, 'r') as fhandle:
        lines = fhandle.readlines()

    elements, xyz = [], []
    for line in lines[2:]:
        parts = line.split()
        # ignores additional columns beyond the first 4
        elements.append(parts[0])
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])

    atomic_numbers = [ELEMENT_TO_ATOMIC_NUMBER[el] for el in elements]
    mult = 1 + Structure.get_unpaired_electrons(atomic_numbers, charge=0)

    def _comment_looks_like_smiles(line):
        # Basic SMILES pattern - atoms, bonds, brackets, rings
        s = line.strip()
        pattern = r'^[A-Za-z0-9\[\]()=#\-+@/\\%.]+$'
        return bool(re.match(pattern, s)) and len(s) > 0

    if not parse_comment:
        return Structure(elements=elements,
                         xyz=np.array(xyz),
                         smiles=lines[1].strip()
                         if _comment_looks_like_smiles(lines[1]) else "",
                         atomic_numbers=atomic_numbers,
                         multiplicity=mult,
                         metadata={'from_xyz_file': str(xyz_path)})
    else:
        # Example comment line:
        # B(=O)OB=O unique_id:54355245865 charge:0 multiplicity:1 energy_hartree:-275.62039364954484
        comment_line = lines[1].strip()

        # Initialize default values
        smiles = ""
        unique_id = None
        charge = 0
        multiplicity = 1
        property = {}

        # Look for key patterns in the comment line
        if "unique_id:" in comment_line:
            # Extract SMILES (everything before 'unique_id:')
            smiles = comment_line.split("unique_id:")[0].strip()

        if uid_match := re.search(r"unique_id:\s*(\S+)", comment_line):
            unique_id = uid_match.group(1)
        if charge_match := re.search(r"charge:\s*(-?\d+)", comment_line):
            charge = int(charge_match.group(1))
        if multiplicity_match := re.search(r"multiplicity:\s*(\d+)",
                                           comment_line):
            multiplicity = int(multiplicity_match.group(1))

        if dft_energy_match := re.search(
                r"energy_hartree:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                comment_line):
            property[DFT_ENERGY_KEY] = float(dft_energy_match.group(1))

        return Structure(elements=elements,
                         xyz=xyz,
                         unique_id=unique_id,
                         smiles=smiles,
                         charge=charge,
                         multiplicity=multiplicity,
                         property=property,
                         metadata={'from_xyz_file': str(xyz_path)})


def write_molecule_property(st_or_sts: StructureType,
                            filename: str,
                            save_metadata=True):
    """
    Write the molecule-level properties structure properties of one or multiple
    Structure objects to a CSV or Parquet file.
    """
    if not filename.endswith(COLUMNAR_FILE_EXTENSIONS):
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    if isinstance(st_or_sts, Structure):
        sts = [st_or_sts]
    else:
        sts = st_or_sts

    # One dict per structure
    data = []
    _mol_keys = [*SHARED_KEYS, 'multiplicity', 'charge']
    for st in sts:
        row = {key: getattr(st, key) for key in _mol_keys}
        row.update(st.property)
        data.append(row)
    df = pd.DataFrame(data)

    if filename.endswith('.csv'):
        df.to_csv(filename, index=False)

    if filename.endswith(('.parquet', '.parq')):
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filename)

    if save_metadata:
        mettadata_list = [st.metadata for st in sts]
        index = [getattr(st, "unique_id") for st in sts]
        metadata_df = pd.DataFrame(mettadata_list, index=index)
        metadata_df.index.name = "unique_id"
        metadata_df.to_csv('metadata.csv', index=True)


def write_atom_property(st_or_sts: StructureType,
                        filename: str,
                        parq_metadata=None):
    """
    Write out the atom-level structure properties to a CSV or Parquet file.
    """

    if not filename.endswith(COLUMNAR_FILE_EXTENSIONS):
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    if isinstance(st_or_sts, Structure):
        sts = [st_or_sts]
    else:
        sts = st_or_sts

    st_dfs = []
    for st in sts:
        data = {
            # Add shared keys: unique_id, smiles
            key: [getattr(st, key) for _ in range(len(st.elements))]
            for key in SHARED_KEYS
        }
        data.update({
            'element': st.elements,
            'x': st.xyz[:, 0],
            'y': st.xyz[:, 1],
            'z': st.xyz[:, 2],
        })
        data.update(st.atom_property)
        st_dfs.append(pd.DataFrame(data))

    df = pd.concat(st_dfs, ignore_index=True)

    if filename.endswith('.csv'):
        df.to_csv(filename, index=False)

    if filename.endswith(('.parquet', '.parq')):
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        if parq_metadata:
            # Add metadata if given
            table = table.replace_schema_metadata(parq_metadata)
        # Write to Parquet file
        pq.write_table(table, filename)
