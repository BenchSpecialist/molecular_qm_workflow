#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Commandline API to run the molecular geometry optimization and property calculation pipeline.

Import the parent directory to the system path to run this script from anywhere.
```
export PATH="/path/to/mqc_pipeline/cmdline:$PATH"
```

Usage:
- Generate a default configuration file:
    $ mqc_runner.py --write-default-config <config.yaml>

- Run the pipeline with a configuration file:
    $ mqc_runner.py --config <config.yaml>

- Batch inputs and write out SLURM scripts, but NOT submit for debugging:
    $ mqc_runner.py --config <config.yaml> --dry-run
"""
import pprint
import pickle
import argparse
import subprocess
from pathlib import Path

from mqc_pipeline.workflow import pipeline
from mqc_pipeline.workflow.io import read_smiles, read_xyz_dir
from mqc_pipeline.settings import PipelineSettings
from mqc_pipeline.validate import validate_input
from mqc_pipeline.util import logger, change_dir

_PYTHON_EXE = "/mnt/filesystem/dev_renkeh/mqc-env/bin/python"

SLURM_CMD = """#!/bin/bash

#SBATCH --job-name={JOB_NAME}_{BATCH_ID}
#SBATCH --output={JOB_LOG}
#SBATCH --error={JOB_LOG}
#SBATCH --nodes=1            # Use 1 node
#SBATCH --ntasks-per-node=1  # CPU spec: Request 1 MPI task/independent process
#SBATCH --gres=gpu:1         # Allocate 1 GPU
#SBATCH --partition=high-priority
#SBATCH --mem=100G           # 1500 GB limit per compute node, CPUs could be oversubscribed based on request mem
#SBATCH --exclude=fs-sn-001,fs-sn-002

echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

srun {PYTHON_EXE} << EOF
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
                        help="Path to the YAML configuration file.")

    parser.add_argument("--write-default-config",
                        type=str,
                        help="Write the default configuration file and exit.")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
        "For debug: Batch inputs, write sbatch script and print command without executing it. "
    )

    return parser.parse_args()


def _count_files(directory) -> int:
    return sum(1 for p in Path(directory).iterdir() if p.is_file())


def _distribute_inputs(num_inputs, num_jobs) -> list[int]:
    base = num_inputs // num_jobs
    remainder = num_inputs % num_jobs
    batch_sizes = [
        base + 1 if i < remainder else base for i in range(num_jobs)
    ]
    return batch_sizes


def main():
    """
    Main function to run the pipeline.
    """
    args = _parse_args()

    if args.write_default_config:
        # Write the default configuration file and exit the program.
        # This is useful for new user to set up the configuration file.
        PipelineSettings.write_default_config_to_yaml(
            args.write_default_config)
        print(
            f"Default configuration file written to {args.write_default_config}"
        )
        return

    if args.config is None:
        raise SystemExit("No configuration file provided.")

    settings = PipelineSettings.from_yaml(args.config)

    # Detailed validation of the input file or directory
    validate_input(settings.input_file_or_dir)

    logger.info(f"Settings:\n{pprint.pformat(dict(settings))}")

    if settings.num_jobs == 0:
        # Run the pipeline locally without SLURM orchestration
        logger.info(
            "Running the pipeline locally without SLURM orchestration.")
        pipeline.run_from_config_settings(settings)
        logger.info("Pipeline finished successfully.")
        return

    input_path = Path(settings.input_file_or_dir)
    _cached_config_path = input_path.parent.resolve() / "_config.pkl"
    # Caching the validated settings for batch jobs via SLURM
    with open(_cached_config_path, "wb") as f:
        pickle.dump(settings, f)

    # Create output directory for batch files and logs
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    submitted_jobs = []

    # Process based on input type
    if input_path.is_file():
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

            # Save batch data to disk
            batch_dir = output_dir / f"batch_{batch_id}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_file = batch_dir / f"input_smiles.pkl"
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_smiles, fh)

            # Write batch info to human-readable format for debugging
            with open(batch_dir / f"_input_smiles.txt", 'w') as fh:
                fh.write("\n".join(batch_smiles))

            # make outputs generated in the batch dir
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

            # Save batch data using pickle
            batch_dir = output_dir / f"batch_{batch_id}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_file = batch_dir / f"input_sts.pkl"
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_sts, fh)

            # Write batch info to human-readable format
            with open(batch_dir / f"_input_xyz.txt", 'w') as fh:
                fh.write("\n".join(
                    [st.metadata.get('from_xyz_file') for st in batch_sts]))

            # make outputs generated in the batch dir
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
    output_dir = Path(config.output_dir).resolve()
    _slurm_sh_dir = output_dir / "slurm_scripts"
    _slurm_sh_dir.mkdir(parents=True, exist_ok=True)
    sbatch_sh_path = _slurm_sh_dir / f"submit_{batch_id}.sh"

    _slum_log_dir = output_dir / "slurm_logs"
    _slum_log_dir.mkdir(parents=True, exist_ok=True)
    job_log = _slum_log_dir / f"{batch_id}.log"

    # Get absolute paths for the batch input file (in pkl format)
    batch_file = Path(batch_file).resolve()

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
        print(f"Dry run: wrote {sbatch_sh_path}")
        return None

    # Submit the job using sbatch
    try:
        result = subprocess.run(["sbatch", str(sbatch_sh_path)],
                                check=True,
                                text=True,
                                capture_output=True)
        job_id = result.stdout.strip().split()[-1]
        msg = f"Submitted batch {batch_id} as job {job_id}"
        logger.info(msg)
        print(msg)
        return job_id
    except subprocess.CalledProcessError as e:
        msg = f"Failed to submit job: {str(e)}"
        logger.error(msg)
        print(msg)
        return None


if __name__ == "__main__":
    main()
