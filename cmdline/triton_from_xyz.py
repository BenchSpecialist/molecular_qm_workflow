#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Cmdline utility to read XYZs from a CSV/Parquet file, run molecular geometry
optimization via Triton inference server and subsequent property calculations
using extended AIMNET2 model.

The input CSV/Parquet file must contain two columns: 'smiles' and 'xyz_block'.
"""
import os
import pickle
import argparse
import warnings
import subprocess
from pathlib import Path
from tabulate import tabulate
import polars

from mqc_pipeline.util import get_default_logger, change_dir

logger = get_default_logger()

# Maximum number of structures per job
# Each request to Triton server takes at most 1024 structures.
# Limit the number of requests per SLURM job to avoid timeout issues.
DEFAULT_MAX_REQUESTS_PER_JOB = 12
MAX_MOLS_PER_JOB = 1024 * int(
    os.environ.get("MAX_REQUESTS_PER_JOB", DEFAULT_MAX_REQUESTS_PER_JOB))

SLURM_CMD = """#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --output={JOB_LOG}
#SBATCH --error={JOB_LOG}
#SBATCH --nodes=1                        # Use 1 node
#SBATCH --ntasks-per-node=1              # Run a single task
#SBATCH --cpus-per-task=128              # Total 128 CPUs available per node
#SBATCH --gres=gpu:8                     # Request 8 GPUs
#SBATCH --partition=high-priority
#SBATCH --mem=800G
#SBATCH --nodelist={NODE_NAME}

# set environment variable for inference URL
export INFERENCE_URL=http://localhost:8000/v1/infer

srun /mnt/filesystem/dev_renkeh/mqc-env/bin/python << EOF
import pickle
from mqc_pipeline.workflow.io import get_sts_from_file
from mqc_pipeline.workflow.triton_pipeline import run_pipeline

# Load the validated pydantic model from cache
with open("{CACHED_CONFIG}", "rb") as f:
    settings = pickle.load(f)

# Load batch data
sts = get_sts_from_file("{INFILE}", start_idx={START_IDX}, batch_size={BATCH_SIZE})

run_pipeline(sts, settings)

EOF
"""


def _parse_args():
    """
    Define command-line interface
    """
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file. ")
    return parser.parse_args()


def main():
    args = _parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mqc_pipeline.workflow import triton_pipeline

    try:
        settings = triton_pipeline.TritonPipelineSettings.from_yaml(
            args.config)
    except Exception as e:
        logger.error(str(e))
        print(str(e))
        raise SystemExit(1)

    active_node_file = Path(settings.active_triton_nodes_file)
    if active_node_file.exists():
        # Read from file and keep unique node names
        nodes_from_file = {
            node_name.strip()
            for node_name in active_node_file.read_text().strip().splitlines()
        }

        # Filter out invalid node names
        active_nodes = [
            node for node in nodes_from_file
            if node.startswith("fs-sn-") and node.split("-")[-1].isdigit()
        ]
        logger.info(f"{len(active_nodes)} active Triton server nodes.")
    else:
        raise SystemExit(f"'{active_node_file}' does not exist.")

    output_dir = Path.cwd()

    _cached_config_path = output_dir / "_config.pkl"
    # Caching config for batch jobs
    with open(_cached_config_path, "wb") as f:
        pickle.dump(settings, f)

    infile = Path(settings.input_file).resolve()
    if infile.suffix not in ['.csv', '.parquet', '.parq']:
        raise SystemExit(
            f"Input file must be a CSV or Parquet file, got: {infile}")

    # Get number of rows without loading all data
    if infile.suffix == '.csv':
        num_inputs = polars.scan_csv(
            str(infile)).collect(streaming=True).height
    else:  # Parquet file
        import pyarrow.parquet as pq
        num_inputs = pq.read_metadata(str(infile)).num_rows

    logger.info(f"{infile} contains {num_inputs} molecules")

    n_full_size_jobs = num_inputs // MAX_MOLS_PER_JOB
    n_remainder_jobs = int(num_inputs % MAX_MOLS_PER_JOB > 0)
    job_distribution = [(MAX_MOLS_PER_JOB, n_full_size_jobs),
                        (num_inputs % MAX_MOLS_PER_JOB, n_remainder_jobs)]

    # Track submitted jobs
    submitted_jobs = []
    node2jobids = {node: [] for node in active_nodes}

    # Distribute jobs across nodes
    start_idx = 0
    node_idx = 0
    for batch_size, num_jobs in job_distribution:
        for _ in range(num_jobs):
            node_name = active_nodes[node_idx]
            # Create a unique directory for each job on this node
            job_dir_name = f"batch_{start_idx}-{start_idx + batch_size-1}"

            batch_dir = output_dir / job_dir_name

            with change_dir(batch_dir):
                job_id = _submit_job(batch_dir=batch_dir,
                                     node_name=node_name,
                                     cached_config=_cached_config_path,
                                     infile=infile,
                                     start_idx=start_idx,
                                     batch_size=batch_size)
            if job_id:
                submitted_jobs.append(
                    (job_dir_name, node_name, job_id, batch_size))
                node2jobids[node_name].append(job_id)

            # Move to next node in round-robin fashion
            node_idx = (node_idx + 1) % len(active_nodes)
            start_idx += batch_size

    ## Print summary of submitted jobs
    node_job_counts = [(node, len(job_ids), ','.join(job_ids))
                       for node, job_ids in node2jobids.items() if job_ids]
    # Sort by nodename
    node_job_counts.sort(key=lambda x: x[0])
    node_jobs_info = tabulate(node_job_counts,
                              headers=['Node', 'Num_jobs', 'Job IDs'],
                              tablefmt="rounded_outline",
                              colalign=("left", "right", "left"))

    job_dir_info = tabulate(
        submitted_jobs,
        headers=['Job Directory', 'Node', 'Job ID', 'Num_SMILES'],
        tablefmt="simple",
        colalign=("left", "left", "left", "right"))
    logger.info(
        f"Submitted {len(submitted_jobs)} jobs:\n{node_jobs_info}\n{job_dir_info}"
    )
    print(
        f"Total {len(submitted_jobs)} jobs on {len(active_nodes)} nodes:\n{node_jobs_info}"
    )


def _submit_job(batch_dir: str, node_name: str, cached_config: str,
                infile: str, start_idx: int, batch_size: int) -> str | None:
    batch_dir = Path(batch_dir).resolve()

    script_path = batch_dir / f"submit.sh"

    job_log_dir = batch_dir.parent / "slurm_logs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    job_log_path = job_log_dir / f"{batch_dir.name}.log"

    # Create the SLURM command
    slurm_cmd = SLURM_CMD.format(
        JOB_NAME=node_name.lstrip('fs-sn-'),
        JOB_LOG=job_log_path,
        NODE_NAME=node_name,
        CACHED_CONFIG=str(cached_config),
        INFILE=str(infile),
        START_IDX=start_idx,
        BATCH_SIZE=batch_size,
    )

    script_path.write_text(slurm_cmd)

    # Submit the job using sbatch
    try:
        result = subprocess.run(["sbatch", str(script_path)],
                                check=True,
                                text=True,
                                capture_output=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted to {node_name}: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        msg = f"Failed to submit job for {node_name}: {str(e)}"
        logger.error(msg)
        print(msg)
        return None


if __name__ == "__main__":
    main()
