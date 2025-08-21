import polars
from pathlib import Path

from ..util import get_default_logger, profiler

logger = get_default_logger()

SCHEMA_OVERRIDES = {'unique_id': polars.Utf8}


@profiler
def combine_csv_files_chunk(outfile_paths: list[Path],
                            mol_out_format='parq',
                            files_per_chunk: int = 120) -> None:
    """
    Combine CSV files with the same name from all batch directories.

    :param outfile_paths: List of output CSV file paths to combine which usually have same names
                          e.g., ["batch_0/molecule_property.csv", "batch_1/molecule_property.csv"]
    :param mol_out_format: Output format for the combined molecule property table
                           and metadata table, either 'csv' or 'parq'.
    :param files_per_chunk: Number of files to process in each chunk (memory management)
    """
    mol_out_format = mol_out_format.lower()
    csv_files = [
        Path(p) for p in outfile_paths if (p.suffix == '.csv' and p.exists())
    ]
    if not csv_files:
        return

    # Chunked processing for memory efficiency
    print("\nUsing chunked processing for combining CSV files.")
    combined_df = _combine_with_chunking(csv_files, chunk_size=files_per_chunk)
    print(f"Columns: {combined_df.columns} ")
    print(f'First 5 rows: {combined_df.head(5)}')

    # Write output
    from mqc_pipeline.util import write_df_to_parq_duckdb

    filename = outfile_paths[0].name
    combined_file = outfile_paths[0].parent.parent / (
        Path(filename).with_suffix('.parquet') if "atom_property" in filename
        or mol_out_format == 'parq' else filename)

    if "atom_property" in filename or mol_out_format == 'parq':
        write_df_to_parq_duckdb(combined_df, combined_file)
    else:
        combined_df.write_csv(combined_file)

    logger.info(f'{combined_df.height} rows in {combined_file}.')
    print(
        f"Combined {len(csv_files)} {filename} files into {combined_file}, {combined_df.height} rows"
    )


def _combine_with_chunking(csv_files: list[Path],
                           chunk_size: int) -> polars.DataFrame:
    """Memory-efficient chunked processing."""
    combined_df = None

    for i in range(0, len(csv_files), chunk_size):
        chunk_files = csv_files[i:i + chunk_size]

        # Read chunk
        chunk_combined = _process_chunk(chunk_files)

        # Combine with previous chunks
        if combined_df is None:
            combined_df = chunk_combined
        else:
            combined_df = polars.concat([combined_df, chunk_combined],
                                        how='diagonal')

        # Optional: Force garbage collection for very large datasets
        import gc
        gc.collect()

        logger.info(
            f"Processed chunk {i//chunk_size + 1}/{(len(csv_files) + chunk_size - 1)//chunk_size}"
        )

    return combined_df


def _process_chunk(chunk_files):
    """
    Process a chunk of CSV files and combine them.

    :param chunk_files: List of CSV file paths to process
    :return: Combined DataFrame from all files in the chunk
    """
    chunk_dfs = []
    float_cols = 'triton_energy_ev homo lumo esp_min esp_max combustion_heat_ev vdw_volume_angstrom3'
    str_cols = ['unique_id', 'smiles', 'canonical_smiles']
    for csv_file in chunk_files:
        df = polars.read_csv(csv_file, schema_overrides=SCHEMA_OVERRIDES)
        # Try to cast string columns to float if they contain numeric data
        for col in df.columns:
            if df[col].dtype == polars.Utf8 and col not in str_cols:
                try:
                    # Check first 10 non-null values to see if they're numeric
                    sample_values = df.select(
                        polars.col(col).drop_nulls().head(
                            10)).to_series().to_list()
                    is_numeric = all(
                        val.replace('.', '').replace('-', '').replace(
                            '+', '').isdigit() for val in sample_values
                        if val and isinstance(val, str))

                    if is_numeric and sample_values:
                        df = df.with_columns(
                            polars.col(col).str.strip_chars().cast(
                                polars.Float64, strict=False))
                except:
                    pass  # Keep as string if casting fails

        chunk_dfs.append(df)

    return polars.concat(chunk_dfs, how='diagonal')


@profiler
def combine_csv_files_parallel(outfile_paths: list[Path],
                               mol_out_format='parq',
                               n_workers: int = None) -> None:
    """
    Combine CSV files with the same name from all batch directories using
    parallel processing
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import gc

    if n_workers is None:
        # 32 cores on fs-s-login-001
        n_workers = min(mp.cpu_count(), 16)

    mol_out_format = mol_out_format.lower()
    csv_files = [
        Path(p) for p in outfile_paths if (p.suffix == '.csv' and p.exists())
    ]
    if not csv_files:
        return

    # Split files into chunks for parallel processing
    chunk_size = max(1, len(csv_files) // n_workers)
    file_chunks = [
        csv_files[i:i + chunk_size]
        for i in range(0, len(csv_files), chunk_size)
    ]

    # Process chunks in parallel
    chunk_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(_process_chunk, chunk) for chunk in file_chunks
        ]

        # Collect results as they complete
        chunk_results = [future.result() for future in as_completed(futures)]

    # Force garbage collection after executor is closed
    gc.collect()

    # Combine all chunks
    try:
        combined_df = polars.concat(chunk_results, how='vertical')
    except polars.exceptions.PolarsError as e:
        # Handle various Polars-specific errors that might occur during concat
        logger.debug(
            f"Vertical concat failed with error: {e}. Trying diagonal concat for large datasets."
        )
        combined_df = polars.concat(chunk_results, how='diagonal')

    print(f"Columns: {combined_df.columns} ")
    print(f'First 5 rows: {combined_df.head(5)}')

    # Free up memory from chunk results
    del chunk_results
    gc.collect()

    # Write output
    from mqc_pipeline.util import write_df_to_parq_duckdb

    filename = outfile_paths[0].name
    # Save combined file to the parent directory of batch directories
    combined_file = outfile_paths[0].parent.parent / (
        Path(filename).with_suffix('.parquet') if "atom_property" in filename
        or mol_out_format == 'parq' else filename)

    if "atom_property" in filename or mol_out_format == 'parq':
        write_df_to_parq_duckdb(combined_df, combined_file)
    else:
        combined_df.write_csv(combined_file)

    logger.info(f'{combined_df.height} rows in {combined_file}.')
    print(
        f"Combined {len(csv_files)} {filename} files into {combined_file}, {combined_df.height} rows"
    )
