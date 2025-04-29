"""
Utilities for uploading Structure objects to Snowflake using Parquet format.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from uuid import uuid4
from tempfile import NamedTemporaryFile
from snowflake.snowpark import Session

from .common import Structure

DEFAULT_BATCH_SIZE = 10000


def batch_structures_to_parquet(structures: list[Structure],
                                compression='snappy') -> bytes:
    """
    Convert multiple Structure objects to compressed Parquet binary data.

    :param structures: List of Structure objects
    :param compression: Compression algorithm (snappy, gzip, zstd)

    :return: Compressed Parquet binary data
    """
    # Convert structures to dictionary of arrays for columnar storage
    data = {
        "unique_id": [],
        "elements": [],
        "xyz_flat": [],  # Store as flattened arrays
        "n_atoms": [],  # Store atom count for reconstruction
        "atomic_numbers": [],
        "smiles": [],
        "charge": [],
        "multiplicity": []
    }

    # Property and metadata dictionaries will be stored as JSON strings
    property_fields = set()
    metadata_fields = set()

    # First pass - collect all property and metadata fields
    for structure in structures:
        if structure.property:
            property_fields.update(structure.property.keys())
        if structure.metadata:
            metadata_fields.update(structure.metadata.keys())

    # Add property and metadata fields to data dictionary
    for field in property_fields:
        data[f"property_{field}"] = []

    for field in metadata_fields:
        data[f"metadata_{field}"] = []

    # Second pass - populate data
    for structure in structures:
        data["unique_id"].append(structure.unique_id)
        data["elements"].append(structure.elements)
        data["n_atoms"].append(len(structure.elements))
        data["xyz_flat"].extend(structure.xyz.flatten())
        data["atomic_numbers"].append(
            structure.atomic_numbers if structure.atomic_numbers else [])
        data["smiles"].append(structure.smiles)
        data["charge"].append(float(structure.charge))
        data["multiplicity"].append(int(structure.multiplicity))

        # Add properties
        for field in property_fields:
            if structure.property and field in structure.property:
                value = structure.property[field]
                # Convert numpy arrays to lists
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                data[f"property_{field}"].append(value)
            else:
                data[f"property_{field}"].append(None)

        # Add metadata
        for field in metadata_fields:
            if structure.metadata and field in structure.metadata:
                value = structure.metadata[field]
                # Convert numpy arrays to lists
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                data[f"metadata_{field}"].append(value)
            else:
                data[f"metadata_{field}"].append(None)

    # Convert to PyArrow Table
    table = pa.Table.from_pydict(data)

    # Write to in-memory Parquet file
    with NamedTemporaryFile() as temp_file:
        pq.write_table(table, temp_file.name, compression=compression)
        with open(temp_file.name, 'rb') as f:
            return f.read()


def upload_structures_in_batches(session: Session,
                                 structures: list[Structure],
                                 table_name: str,
                                 batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """
    Upload structures in batches for efficient data transfer

    :param session: Snowflake session
    :param structures: List of Structure objects
    :param table_name: Target table name
    :param batch_size: Number of structures per batch
    """
    total_structures = len(structures)

    for i in range(0, total_structures, batch_size):
        batch = structures[i:min(i + batch_size, total_structures)]

        # Convert batch to Parquet binary
        parquet_data = batch_structures_to_parquet(batch)

        # Use Snowflake's COPY command with internal staging
        with NamedTemporaryFile(suffix='.parquet') as temp_file:
            temp_file.write(parquet_data)
            temp_file.flush()

            # Create a temporary internal stage
            stage_name = f"TEMP_STAGE_{uuid4().hex[:8]}"
            session.sql(f"CREATE TEMPORARY STAGE {stage_name}").collect()

            # PUT file to stage
            session.file.put(temp_file.name,
                             f"@{stage_name}",
                             auto_compress=False)

            # COPY into table
            session.sql(f"""
                COPY INTO {table_name}
                FROM @{stage_name}
                FILE_FORMAT = (TYPE = PARQUET)
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
            """).collect()

            # Clean up stage
            session.sql(f"DROP STAGE IF EXISTS {stage_name}").collect()
