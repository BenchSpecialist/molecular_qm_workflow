#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Import the parent directory to the system path to run this script from anywhere.
```
export PATH="/path/to/mqc_pipeline/cmdline:$PATH"
```
"""
import pprint
import pickle
import logging
import argparse
import subprocess
from pathlib import Path

from mqc_pipeline.workflow import pipeline
from mqc_pipeline.workflow.io import read_smiles, read_xyz_dir
from mqc_pipeline.settings import PipelineSettings

SLURM_CMD = """#!/bin/bash

#SBATCH --job-name={job_name}_{batch_id}
#SBATCH --output={job_log}
#SBATCH --error={job_log}
#SBATCH --ntasks=1                                 # CPU specification
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --time=24:00:00
#SBATCH --partition=high-priority
#SBATCH --exclude=fs-sn-001,fs-sn-002

{python_exe} << EOF
import pickle
from mqc_pipeline.settings import PipelineSettings
from mqc_pipeline.workflow.pipeline import run_one_batch

# Setup pydantic model directly to save time
settings_dict = {settings_str}

# Load batch data
with open("{batch_file}", "rb") as f:
    inputs = pickle.load(f)

run_one_batch(inputs, settings)

EOF

"""
PYTHON_EXE = "/mnt/filesystem/dev_renkeh/mqc-env/bin/python"


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
        return

    if args.config is None:
        raise SystemExit("No configuration file provided.")

    # Load and validate settings from the configuration file
    logging.info(f"Using configuration file: {args.config}")
    settings = PipelineSettings.from_yaml(args.config)
    logging.info(f"Settings:\n{pprint.pformat(dict(settings))}")

    if settings.num_jobs == 0:
        # Run the pipeline locally without SLURM orchestration
        logging.info(
            "Running the pipeline locally without SLURM orchestration.")
        pipeline.run_from_config_settings(settings)
        logging.info("Pipeline finished successfully.")
        return

    # Create output directory for batch files and logs
    output_dir = Path.cwd() / f"{settings.job_name}_batches"
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.resolve()
    logging.info(f"Output directory: {output_dir}")

    input_path = Path(settings.input_file_or_dir)
    submitted_jobs = []

    # Process based on input type
    if input_path.is_file():
        # Inputs are SMILES strings
        logging.info(f"Reading SMILES from {input_path}")
        smiles_list = read_smiles(input_path)
        batch_sizes = _distribute_inputs(len(smiles_list), settings.num_jobs)

        # Create batches
        start_idx = 0
        for batch_id, batch_size in enumerate(batch_sizes):
            batch_smiles = smiles_list[start_idx:start_idx + batch_size]
            start_idx += batch_size

            # Save batch data using pickle
            batch_file = output_dir / f"batch_{batch_id}.pkl"
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_smiles, fh)

            # Submit SLURM job for this batch
            job_id = _submit_one_slurm_job(settings, batch_id, batch_file,
                                           output_dir)
            if job_id:
                submitted_jobs.append(job_id)

    if input_path.is_dir():
        # Starting from XYZ files
        logging.info(f"Reading XYZ files from {input_path}")
        batch_sizes = _distribute_inputs(_count_files(input_path),
                                         settings.num_jobs)

        # Create structure batches from `read_xyz_dir` generator
        start_idx = 0
        for batch_id, batch_size in enumerate(batch_sizes):
            start_idx += batch_size
            batch_sts = [read_xyz_dir(input_path) for _ in range(batch_size)]

            # Save batch data using pickle
            batch_file = output_dir / f"batch_{batch_id}.pkl"
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_sts, fh)

            # Submit SLURM job for this batch
            job_id = _submit_one_slurm_job(settings, batch_id, batch_file,
                                           output_dir)
            if job_id:
                submitted_jobs.append(job_id)

    logging.info(f"Submitted {len(submitted_jobs)} jobs")
    print(
        f"Submitted {len(submitted_jobs)} jobs with IDs: {', '.join(submitted_jobs)}"
    )


def _submit_one_slurm_job(config: PipelineSettings, batch_id: int,
                          batch_file: str | Path, output_dir: str | Path):
    """
    Launch a single sbatch job.

    :param settings: Pipeline settings
    :param batch_id: Batch identifier number
    :param batch_file: Path to pickle file containing the batch data
    :param output_dir: Directory to store the job script and log

    """
    output_dir = Path(output_dir).resolve()
    script_path = output_dir / f"batch_{batch_id}.sh"
    job_log = output_dir / f"batch_{batch_id}.log"

    # Get absolute paths for the script and log files
    batch_file = Path(batch_file).resolve()
    config.input_file_or_dir = str(batch_file)

    # Create the SLURM command
    slurm_cmd = SLURM_CMD.format(job_name=config.job_name,
                                 batch_id=batch_id,
                                 job_log=job_log,
                                 python_exe=PYTHON_EXE,
                                 settings_str=str(config.model_dump()),
                                 batch_file=str(batch_file))

    script_path.write_text(slurm_cmd)

    # Submit the job using sbatch
    try:
        result = subprocess.run(["sbatch", str(script_path)],
                                check=True,
                                text=True,
                                capture_output=True)
        job_id = result.stdout.strip().split()[-1]
        logging.info(f"Submitted batch {batch_id} as job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        print(f"Failed to submit job: {e}")
        return None


if __name__ == "__main__":
    main()
