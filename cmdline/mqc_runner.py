#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Commandline API to run the molecular geometry optimization and property calculation pipeline.

Configure environment variables to allow this script to be executed from any location.
```
export PATH="/path/to/mqc_pipeline/cmdline:$PATH"
export PYTHONPATH="/path/to/mqc_pipeline:$PYTHONPATH"
```

Usage:
- Generate a default configuration file:
    $ mqc_runner.py --write-default-config <config.yaml>

- Run the pipeline with a configuration file:
    $ mqc_runner.py --config <config.yaml>

- Batch inputs and write out SLURM scripts, but NOT submit for debugging:
    $ mqc_runner.py --config <config.yaml> --dry-run
"""
import os
import shutil
import pprint
import pickle
import argparse
import warnings
import subprocess
import polars
from pathlib import Path

from mqc_pipeline.settings import PipelineSettings
from mqc_pipeline.validate import validate_input
from mqc_pipeline.util import get_default_logger, change_dir

logger = get_default_logger()

_PYTHON_EXE = "/mnt/filesystem/dev_renkeh/mqc-env/bin/python"

_SLURM_SH_DIR = "slurm_scripts"
_SLURM_LOG_DIR = "slurm_logs"
_CACHED_CONFIG = "_config.pkl"
_BATCH_SMILES_PKL = "_input_smiles.pkl"
_BATCH_STS_PKL = "_input_sts.pkl"

OUT_FORMATS = ['csv', 'parq']

# Output files in batch directories
OUTFILES = (
    "molecule_property.csv",
    "atom_property.csv",
    "metadata.csv",
    # additional outputs when fluoride_bde is enabled
    "defluorined_molecule_property.csv",
    "defluorined_atom_property.csv")

SLURM_CMD = """#!/bin/bash

#SBATCH --job-name={JOB_NAME}_{BATCH_ID}
#SBATCH --output={JOB_LOG}
#SBATCH --error={JOB_LOG}
#SBATCH --nodes=1            # Use 1 node
#SBATCH --ntasks-per-node=1  # 1 MPI task/independent process
#SBATCH --cpus-per-task=16   # Total 128 CPUs available per node
#SBATCH --gres=gpu:1         # Allocate 1 GPU
#SBATCH --partition=high-priority
#SBATCH --mem=100G           # 1500 GB limit per compute node, CPUs could be oversubscribed based on request mem
#SBATCH --exclude=fs-sn-001,fs-sn-002

echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

srun --cpu-bind=none {PYTHON_EXE} << EOF
import pickle
from mqc_pipeline.workflow.pipeline import run_one_batch

# Load the validated pydantic model from cache
with open("{CACHED_CONFIG}", "rb") as f:
    config = pickle.load(f)

# Load batch data
with open("{BATCH_INPUT}", "rb") as f:
    inputs = pickle.load(f)

run_one_batch(inputs, config)

EOF

"""


def _parse_args():
    """
    Define command-line interface
    """
    parser = argparse.ArgumentParser(
        description=
        "Cmdline utility to run molecular geometry optimization and property calculation pipeline."
    )
    parser.add_argument("--config",
                        type=str,
                        metavar="YAML_FILE",
                        help="Path to the YAML configuration file.")

    parser.add_argument("--write-default-config",
                        type=str,
                        metavar="YAML_FILE",
                        help="Write the default configuration file and exit.")

    parser.add_argument(
        "--validate-input",
        action="store_true",
        help=
        "Validate the input file or directory. This flag needs to be used with --config option."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
        "For debug: Batch inputs, write sbatch script and print command without executing it. "
    )

    parser.add_argument(
        "--combine-results",
        type=str,
        metavar='CSV_OR_PARQUET',
        choices=OUT_FORMATS,
        help=
        "Combine results from all batches into a new file with the specified format. "
        "Supported formats: 'csv', 'parq'. ")

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove temporary files after combining results. "
        "Deletes batch directories, SLURM scripts/logs, and cached config. "
        "Use this to clean workspace after successful pipeline completion. "
        "Batch directories are only removed when this flag is used with --combine-results "
        "to prevent accidental deletion of batch results before they are combined into final output files."
    )

    parser.add_argument(
        "--extract-xyz",
        type=str,
        nargs=2,
        metavar=("ATOM_PROP_FILE", "XYZ_DIR"),
        help=
        "Extract XYZ files from the specified atom_property file to the given directory."
    )

    parser.add_argument(
        "--extended-xyz",
        action="store_true",
        help="Use extended XYZ format for output XYZ files. "
        "This flag is only relevant when --extract-xyz is used.")
    return parser.parse_args()


def _count_files(directory) -> int:
    return sum(1 for p in Path(directory).iterdir()
               if p.is_file() and p.suffix == '.xyz')


def _distribute_inputs(num_inputs, num_jobs) -> list[int]:
    base = num_inputs // num_jobs
    remainder = num_inputs % num_jobs
    batch_sizes = [
        base + 1 if i < remainder else base for i in range(num_jobs)
    ]
    return batch_sizes


def _combine_csv_files(batch_dirs: list[Path],
                       filename: str,
                       mol_out_format='parq') -> None:
    """
    Combine CSV files with the same name from all batch directories.

    :param output_dir: List of batch subdirectories
    :param filename: Name of the CSV file to combine (e.g., "molecule_property.csv")
    :param mol_out_format: Output format for the combined molecule property table
                           and metadata table, either 'csv' or 'parq'.
    """
    mol_out_format = mol_out_format.lower()
    csv_files = [
        batch_dir / filename for batch_dir in batch_dirs
        if (batch_dir / filename).exists()
    ]
    if not csv_files:
        return

    # Read and combine all CSV files
    schema_overrides = {'unique_id': polars.datatypes.Utf8}
    dataframes = [
        polars.read_csv(csv_file, schema_overrides=schema_overrides)
        for csv_file in csv_files
    ]
    combined_df = polars.concat(dataframes, how='diagonal')

    from mqc_pipeline.util import write_df_to_parq_duckdb

    # Write combined df in the parent directory
    if "atom_property" in filename or (mol_out_format == 'parq'):
        # Always save combined atom table in the compact parquet format as it's much larger
        # than molecule_property and metadata table
        output_file = batch_dirs[0].parent / Path(filename).with_suffix(
            '.parquet')
        write_df_to_parq_duckdb(combined_df, output_file)
    else:
        output_file = batch_dirs[0].parent / filename
        combined_df.write_csv(output_file)

    logger.info(f'{combined_df.height} rows in {output_file}.')
    print(
        f"Combined {len(csv_files)} {filename} files into {output_file}, {combined_df.height} rows"
    )


def main():
    """
    Main function to run the pipeline.
    """
    args = _parse_args()

    if args.extract_xyz:
        atom_csv = Path(args.extract_xyz[0]).resolve()
        if not atom_csv.exists():
            raise SystemExit(f"{atom_csv} NOT FOUND.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mqc_pipeline.workflow.io import write_xyz_dir_from_csv

        out_xyz_dir = Path(args.extract_xyz[1]).resolve()
        num_xyz = write_xyz_dir_from_csv(csv_path=atom_csv,
                                         output_dir=out_xyz_dir,
                                         extended_xyz=args.extended_xyz)
        print(f"Whether to use extended XYZ format: {args.extended_xyz}")
        print(f"Extracted {num_xyz} XYZ files to {out_xyz_dir}")
        return

    if args.cleanup:
        output_dir = Path(os.environ.get("MQC_OUTPUT_DIR",
                                         Path.cwd())).resolve()
        # Cleanup SLURM logs and cached pickle files
        for pkl_file in (_CACHED_CONFIG, _BATCH_SMILES_PKL, _BATCH_STS_PKL):
            pkl_path = output_dir / pkl_file
            if pkl_path.exists():
                pkl_path.unlink(missing_ok=True)
                print(f"Removed {pkl_path}")
        dirs_to_remove = [
            d for d in (output_dir / _SLURM_LOG_DIR, ) if d.exists()
        ]
        for dir in dirs_to_remove:
            try:
                shutil.rmtree(dir)
                print(f"Removed {dir}")
            except OSError as e:
                print(f"Failed to remove {dir}: {str(e)}")
        if not args.combine_results:
            return

    if out_format := args.combine_results:
        output_dir = Path(os.environ.get("MQC_OUTPUT_DIR",
                                         Path.cwd())).resolve()

        batch_dirs = [
            d for d in output_dir.iterdir() if d.is_dir() and (
                d.name.startswith("batch_") or d.name.startswith("fs-sn-"))
        ]
        if batch_dirs:
            for outfile in OUTFILES:
                _combine_csv_files(batch_dirs,
                                   outfile,
                                   mol_out_format=out_format)

            # Combine "FAILED_INPUTS.txt" files
            num_failed_input_files = sum(1 for batch_dir in batch_dirs
                                         if (batch_dir /
                                             "FAILED_INPUTS.txt").exists())
            cmd = [
                'find', '.', '-type', 'f', '-name', 'FAILED_INPUTS.txt',
                '-exec', 'cat', '{}', ';'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if len(result.stdout) > 0:
                combined_txt = output_dir / 'all_failed_inputs.txt'
                with open(combined_txt, 'w') as f:
                    f.write(result.stdout)
                print(
                    f"Combined {num_failed_input_files} FAILED_INPUTS.txt into {combined_txt}"
                )

        if args.cleanup:
            for dir in batch_dirs:
                try:
                    shutil.rmtree(dir)
                    print(f"Removed {dir}")
                except OSError as e:
                    print(f"Failed to remove {dir}: {str(e)}")
            print('Cleanup completed.')
        return

    if args.write_default_config:
        # Write the default configuration file and exit the program.
        # This is useful for new user to set up the configuration file.
        PipelineSettings.write_default_config_to_yaml(
            args.write_default_config)
        msg = f"Default configuration file written to {args.write_default_config}"
        print(msg)
        return

    if args.config is None:
        raise SystemExit("No configuration file provided.")

    try:
        settings = PipelineSettings.from_yaml(args.config)
    except Exception as e:
        print(str(e))
        raise SystemExit(1)

    if args.validate_input:
        try:
            # Detailed validation of the input file or directory
            validate_input(settings.input_file_or_dir)
            print("Input validation successful.")
        except Exception as e:
            print(f"Input validation failed: {str(e)}")
            raise SystemExit(1)

    logger.info(f"Settings:\n{pprint.pformat(dict(settings))}")

    if settings.num_jobs == 0:
        # Import here to speed up the startup time of the script
        from mqc_pipeline.workflow.pipeline import run_from_config_settings
        # Run the pipeline locally without SLURM orchestration
        logger.info(
            "Running the pipeline locally without SLURM orchestration.")
        run_from_config_settings(settings)
        logger.info("Pipeline finished successfully.")
        return

    # Import here to speed up the startup time of the script
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mqc_pipeline.workflow.io import read_smiles, read_xyz_dir

    input_path = Path(settings.input_file_or_dir)
    _cached_config_path = input_path.parent.resolve() / _CACHED_CONFIG
    # Caching the validated settings for batch jobs via SLURM
    with open(_cached_config_path, "wb") as f:
        pickle.dump(settings, f)

    # Create output directory for batch files and logs
    output_dir = Path(settings.output_dir).resolve()
    # Cache path to an environment variable so it's accessible in the workflow
    # postprocessing when using "--combine-results" and "--cleanup" flags
    os.environ["MQC_OUTPUT_DIR"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submitted_jobs = []

    are_smiles_inputs = input_path.suffix in (
        '.txt', '.csv') or (input_path.suffix in ('.pkl', '.pickle')
                            and settings.pickled_input_type == 'smiles')

    # Process based on input type
    if are_smiles_inputs:
        # Inputs are SMILES strings
        smiles_list = read_smiles(input_path)
        batch_sizes = _distribute_inputs(len(smiles_list), settings.num_jobs)
        msg = (
            f"Read {len(smiles_list)} SMILES from {input_path}\n"
            f"Requesting {settings.num_jobs} jobs, batch sizes: {batch_sizes}")
        logger.info(msg)
        print(msg)

        # Create batches
        start_idx = 0
        for batch_id, batch_size in enumerate(batch_sizes):
            batch_smiles = smiles_list[start_idx:start_idx + batch_size]
            start_idx += batch_size

            # Determine output directory: use output_dir directly for single job
            if settings.num_jobs == 1:
                batch_dir = output_dir
            else:
                batch_dir = output_dir / f"batch_{batch_id}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                with open(batch_dir / f"batch_{batch_id}_smiles.txt",
                          'w') as fh:
                    fh.write("\n".join(batch_smiles))

            batch_file = batch_dir / _BATCH_SMILES_PKL
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_smiles, fh)

            # Make outputs generated in the batch_dir
            with change_dir(batch_dir):
                # Submit SLURM job for this batch
                job_id = _submit_one_slurm_job(
                    config=settings,
                    batch_id=batch_id,
                    batch_file=batch_file,
                    cached_config=_cached_config_path,
                    dry_run=args.dry_run)
            if job_id:
                submitted_jobs.append(job_id)

    ########################################
    if input_path.is_dir():
        # Starting from XYZ files
        num_xyz_files = _count_files(input_path)
        batch_sizes = _distribute_inputs(num_xyz_files, settings.num_jobs)
        msg = (
            f"Read {num_xyz_files} XYZ files from {input_path}\n"
            f"Requesting {settings.num_jobs} jobs, batch sizes: {batch_sizes}")
        logger.info(msg)
        print(msg)

        # Create structure batches from `read_xyz_dir` generator
        st_generator = read_xyz_dir(input_path)
        for batch_id, batch_size in enumerate(batch_sizes):
            batch_sts = [next(st_generator) for _ in range(batch_size)]

            # Determine output directory: use output_dir directly for single job
            if settings.num_jobs == 1:
                batch_dir = output_dir
            else:
                batch_dir = output_dir / f"batch_{batch_id}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                # Write batch info to human-readable format for debugging
                with open(batch_dir / "_input_xyz_paths.txt", 'w') as fh:
                    fh.write("\n".join([
                        st.metadata.get('from_xyz_file') for st in batch_sts
                    ]))

            batch_file = batch_dir / _BATCH_STS_PKL
            # Serialize Structure objects in the current batch using pickle
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_sts, fh)

            # Make outputs generated in the batch dir
            with change_dir(batch_dir):
                # Submit SLURM job for this batch
                job_id = _submit_one_slurm_job(
                    config=settings,
                    batch_id=batch_id,
                    batch_file=batch_file,
                    cached_config=_cached_config_path,
                    dry_run=args.dry_run)
            if job_id:
                submitted_jobs.append(job_id)

    if input_path.suffix in (
            '.pkl', '.pickle') and settings.pickled_input_type == 'structure':
        # Starting from pickled Structure objects
        with open(input_path, 'rb') as fh:
            all_sts = pickle.load(fh)
        batch_sizes = _distribute_inputs(len(all_sts), settings.num_jobs)
        msg = (
            f"Read {len(all_sts)} structure objects from {input_path}\n"
            f"Requesting {settings.num_jobs} jobs, batch sizes: {batch_sizes}")
        logger.info(msg)
        print(msg)

        # Create batches
        start_idx = 0
        for batch_id, batch_size in enumerate(batch_sizes):
            batch_sts = all_sts[start_idx:start_idx + batch_size]
            start_idx += batch_size

            # Determine output directory: use output_dir directly for single job
            if settings.num_jobs == 1:
                batch_dir = output_dir
            else:
                batch_dir = output_dir / f"batch_{batch_id}"
                batch_dir.mkdir(parents=True, exist_ok=True)

            batch_file = batch_dir / _BATCH_STS_PKL
            # Serialize Structure objects in the current batch using pickle
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_sts, fh)

            # Make outputs generated in the batch_dir
            with change_dir(batch_dir):
                # Submit SLURM job for this batch
                job_id = _submit_one_slurm_job(
                    config=settings,
                    batch_id=batch_id,
                    batch_file=batch_file,
                    cached_config=_cached_config_path,
                    dry_run=args.dry_run)
            if job_id:
                submitted_jobs.append(job_id)

    msg = f"Submitted {len(submitted_jobs)} jobs with IDs: {', '.join(submitted_jobs)}"
    logger.info(msg)
    print(msg)


def _submit_one_slurm_job(config: PipelineSettings,
                          batch_id: int,
                          batch_file: str | Path,
                          cached_config: str | Path,
                          dry_run: bool = False):
    """
    Launch a single sbatch job.
    """
    # Get absolute paths for the batch input file (in pkl format)
    batch_file = Path(batch_file).resolve()
    sbatch_sh_path = batch_file.parent / f"submit.sh"

    output_dir = Path(config.output_dir).resolve()
    job_log_dir = output_dir / _SLURM_LOG_DIR
    job_log_dir.mkdir(parents=True, exist_ok=True)
    job_log = job_log_dir / f"{batch_file.parent.name}.log"

    # Create the SLURM command
    slurm_cmd = SLURM_CMD.format(JOB_NAME=config.job_name,
                                 BATCH_ID=batch_id,
                                 JOB_LOG=job_log,
                                 PYTHON_EXE=_PYTHON_EXE,
                                 CACHED_CONFIG=str(cached_config),
                                 BATCH_INPUT=str(batch_file))

    sbatch_sh_path.write_text(slurm_cmd)

    if dry_run:
        # Print the command instead of executing it
        msg = f"Dry run: wrote {sbatch_sh_path}"
        logger.info(msg)
        print(msg)
        return None

    # Submit the job using sbatch
    try:
        result = subprocess.run(["sbatch", str(sbatch_sh_path)],
                                check=True,
                                text=True,
                                capture_output=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {str(e)}")
        return None


if __name__ == "__main__":
    main()
