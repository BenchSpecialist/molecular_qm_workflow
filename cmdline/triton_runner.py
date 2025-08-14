#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Cmdline utility to run molecular geometry optimization via Triton inference server,
and subsequent property calculations using extended AIMNET2 model or DFT.

Configure environment variables to allow this script to be executed from any location.
```
export PATH="/path/to/mqc_pipeline/cmdline:$PATH"
export PYTHONPATH="/path/to/mqc_pipeline:$PYTHONPATH"
```

Please follow these steps strictly to run the Triton pipeline:
1. Generate `ACTIVE_TRITON_NODES.txt` file that contains available Triton server nodes by:
    $ triton_runner.py --get-active-triton-nodes

2. Estimate your workload and request the total number of Triton server nodes to run:
    $ triton_runner.py --request-num-server-nodes <num_active_nodes>
   Please wait for at least 2 minutes for all servers to start.

3. Generate a default configuration file:
    $ triton_runner.py --write-default-config <config.yaml>

4. Run the pipeline with the configuration file:
    $ triton_runner.py --config <config.yaml>

Other options:
- To set Triton servers on specific node IDs:
    $ triton_runner.py --set-server-on-node-ids 1,2,63,64
"""
import os
import pickle
import argparse
import warnings
import subprocess
from pathlib import Path
from tabulate import tabulate

from mqc_pipeline.util import get_default_logger, change_dir

logger = get_default_logger()

_PYTHON_EXE = "/mnt/filesystem/dev_renkeh/mqc-env/bin/python"

_ACTIVE_NODES_FILE = "ACTIVE_TRITON_NODES.txt"

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

srun {PYTHON_EXE} << EOF
import pickle
from mqc_pipeline.workflow.triton_pipeline import run_pipeline

# Load the validated pydantic model from cache
with open("{CACHED_CONFIG}", "rb") as f:
    settings = pickle.load(f)

# Load batch data
with open("{BATCH_SMILES_FILE}", "rb") as f:
    smiles_list = pickle.load(f)

run_pipeline(smiles_list, settings)

EOF
"""


def _parse_args():
    """
    Define command-line interface
    """
    parser = argparse.ArgumentParser(usage=__doc__)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--request-num-server-nodes",
        type=int,
        metavar="NUM_SERVER_NODES",
        help="Specify the desired number of Triton server nodes to run. "
        "If requested nodes > active nodes: starts additional servers to reach the target; "
        "If requested nodes < active nodes: uses all currently active nodes; "
        f"active nodes are tracked in: {_ACTIVE_NODES_FILE}.")

    group.add_argument(
        "--set-server-on-node-ids",
        type=str,
        metavar="NODE_IDS",
        help=
        "Set Triton servers on specified node IDs (comma-separated, e.g., '32,4,5')"
    )

    group.add_argument("--config",
                       type=str,
                       help="Path to the YAML configuration file. ")

    parser.add_argument(
        "--get-active-triton-nodes",
        action="store_true",
        help=
        f"Get list of active Triton server nodes and write to {_ACTIVE_NODES_FILE}."
    )

    group.add_argument(
        "--stop-server-nodes-in-file",
        type=str,
        metavar="NODE_NAMES_FILE",
        default=None,
        help=
        f"Path to the text file with node names to stop: {_ACTIVE_NODES_FILE}")

    parser.add_argument("--write-default-config",
                        type=str,
                        help="Write the default configuration file and exit.")

    return parser.parse_args()


def _distribute_inputs_with_max_size(
        num_inputs: int,
        num_nodes: int,
        max_size: int = MAX_MOLS_PER_JOB) -> list[tuple[int, int]]:
    """
    Distribute inputs across nodes for batch processing with a maximum batch size limit.

    :param num_inputs: Total number of input items to process
    :param num_nodes: Number of available nodes
    :param max_size: Maximum number of inputs per job

    :return: List of (batch_size, num_jobs_for_this_size) tuples
    """
    total_jobs = (num_inputs + max_size - 1) // max_size
    base_jobs_per_node = total_jobs // num_nodes
    nodes_with_extra = total_jobs % num_nodes

    distribution = []

    # Nodes with base number of jobs
    if base_jobs_per_node > 0:
        distribution.append(
            (max_size, base_jobs_per_node * (num_nodes - nodes_with_extra)))

    # Nodes with one extra job
    if nodes_with_extra > 0:
        distribution.append(
            (max_size, (base_jobs_per_node + 1) * nodes_with_extra))

    # Handle the last partial job
    remaining = num_inputs % max_size
    if remaining > 0:
        # Replace one full job with a partial job
        if distribution:
            last_batch_size, last_count = distribution[-1]
            if last_count > 1:
                distribution[-1] = (last_batch_size, last_count - 1)
                distribution.append((remaining, 1))
            else:
                distribution[-1] = (remaining, 1)
        else:
            distribution.append((remaining, 1))

    count = sum(batch_size * n for batch_size, n in distribution)
    if count != num_inputs:
        raise SystemExit(
            f"_distribute_inputs_with_max_size: Total input count {count} does not match expected {num_inputs}"
        )
    logger.debug(
        f"Distributed {num_inputs} inputs across {num_nodes} nodes with max size {max_size}: {distribution}"
    )
    return distribution


def main():
    args = _parse_args()
    active_node_file = Path(_ACTIVE_NODES_FILE).resolve()

    from mqc_pipeline.workflow import triton_server_util
    if args.get_active_triton_nodes:
        print(
            f'WARNING: `triton_runner.py --get-active-triton-nodes` must be run on fs-s-login-001. '
            'Running it on compute nodes always return empty list.')

        active_nodes = triton_server_util.get_active_server_nodes()
        active_node_file.write_text("\n".join(active_nodes))
        print(f"Number of nodes host Triton servers: {len(active_nodes)}")
        print(f"Active Triton server nodes: {active_nodes}")
        print(f"Wrote {active_node_file}.")
        return

    if node_names_file := args.stop_server_nodes_in_file:
        nodes_to_stop = [
            node
            for node in Path(node_names_file).read_text().strip().splitlines()
            if node.strip()  # Filter out empty lines
        ]

        stopped_nodes = triton_server_util.stop_server_on_nodes(nodes_to_stop)
        print(f"Stopped Triton server nodes: {stopped_nodes}")
        return

    if args.set_server_on_node_ids:
        from mqc_pipeline.workflow.triton_server_util import FS_NODENAME_TEMPLATE
        node_ids = [
            int(node_id.strip())
            for node_id in args.set_server_on_node_ids.split(',')
            if node_id.strip()
        ]
        node_names = [
            FS_NODENAME_TEMPLATE.format(NODE_ID=node_id)
            for node_id in node_ids
        ]

        # Filter out nodes that are already running Triton server
        active_nodes = triton_server_util.get_active_server_nodes()
        nodes_to_start = [
            name for name in node_names if name not in active_nodes
        ]

        if nodes_to_start:
            triton_server_util.start_server_on_nodes(nodes_to_start)
            print(f"Started Triton server on nodes: {nodes_to_start}")

        if already_active := [
                name for name in node_names if name in active_nodes
        ]:
            print(
                f"Triton server is already running on nodes: {already_active}")

        # Update active nodes file
        active_nodes.extend(nodes_to_start)
        if active_nodes:
            active_nodes.sort(reverse=True)
            active_node_file.write_text("\n".join(active_nodes))
            print(f"Updated {active_node_file} with all active nodes.")
        return

    if total_nodes_requested := args.request_num_server_nodes:
        active_nodes = triton_server_util.get_active_server_nodes()
        remain_n_nodes = total_nodes_requested - len(active_nodes)

        if remain_n_nodes <= 0:
            num_active_nodes = len(active_nodes)
        else:
            num_active_nodes = total_nodes_requested
            logger.info(
                f"Starting {remain_n_nodes} additional Triton server nodes to reach target of {num_active_nodes}."
            )
            available_nodes = list(
                set(triton_server_util.FS_NODE_NAMES) - set(active_nodes))
            # Start servers on nodes with the largest node IDs first.
            # This helps reduce conflicts with other SLURM jobs, since SLURM allocates nodes
            # in alphabetical order. Using the higher-sorted (often less-used) nodes first
            # minimizes the chance of resource competing with other jobs.
            available_nodes.sort(reverse=True)
            new_nodes = available_nodes[:remain_n_nodes]
            active_nodes.extend(new_nodes)
            triton_server_util.start_server_on_nodes(node_names=new_nodes)

        active_node_file.write_text("\n".join(active_nodes))
        print(f"{len(active_nodes)} Triton server nodes: {active_nodes}")
        print(f"Wrote {active_node_file}.")
        return

    if args.write_default_config:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mqc_pipeline.workflow import triton_pipeline

        # Write the default configuration file and exit the program.
        triton_pipeline.TritonPipelineSettings.write_default_config_to_yaml(
            args.write_default_config)
        msg = f"Default configuration file written to {args.write_default_config}"
        print(msg)
        return

    if args.config:
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

        if not active_node_file.exists():
            active_nodes = triton_server_util.get_active_server_nodes()
            if len(active_nodes) == 0:
                raise SystemExit(
                    f"No active Triton server nodes found. Please start servers first by running: "
                    "triton_runner.py --request-num-server-nodes <num_active_nodes>"
                )
        else:
            # Read from file and keep unique node names
            nodes_from_file = {
                node_name.strip()
                for node_name in
                active_node_file.read_text().strip().splitlines()
            }

            # Filter out invalid node names
            active_nodes = [
                node for node in nodes_from_file
                if node.startswith("fs-sn-") and node.split("-")[-1].isdigit()
            ]
            logger.info(f"{len(active_nodes)} active Triton server nodes.")

        from mqc_pipeline.workflow.io import read_smiles

        output_dir = Path.cwd()

        _cached_config_path = output_dir / "_config.pkl"
        # Caching config for batch jobs
        with open(_cached_config_path, "wb") as f:
            pickle.dump(settings, f)

        # TODO: this reads all SMILES strings into memory; may not be suitable
        # for large datasets
        smiles_list = read_smiles(settings.input_file)
        # Create jobs based on input size, available nodes, and max batch size
        job_distribution = _distribute_inputs_with_max_size(
            num_inputs=len(smiles_list),
            num_nodes=len(active_nodes),
            max_size=MAX_MOLS_PER_JOB)

        # Track submitted jobs
        submitted_jobs = []
        start_idx = 0
        node_job_counts = {node: 0 for node in active_nodes}

        # Distribute jobs across nodes
        node_idx = 0
        for batch_size, num_jobs in job_distribution:
            for _ in range(num_jobs):
                node_name = active_nodes[node_idx]
                batch_smiles = smiles_list[start_idx:start_idx + batch_size]
                start_idx += batch_size

                # Create a unique directory for each job on this node
                # job_idx = node_job_counts[node_name]
                # job_dir_name = f"{node_name}_{job_idx}"
                node_job_counts[node_name] += 1

                job_dir_name = f"batch_{start_idx}-{start_idx + batch_size-1}"
                batch_dir = output_dir / job_dir_name

                batch_dir.mkdir(parents=True, exist_ok=True)

                batch_file = batch_dir / "input_smiles.pkl"
                with open(batch_file, 'wb') as fh:
                    pickle.dump(batch_smiles, fh)

                with change_dir(batch_dir):
                    job_id = _submit_job(
                        output_dir=output_dir,
                        node_name=node_name,
                        cached_config=str(_cached_config_path),
                        batch_file=str(batch_file))
                if job_id:
                    submitted_jobs.append(
                        (job_dir_name, node_name, job_id, len(batch_smiles)))

                # Move to next node in round-robin fashion
                node_idx = (node_idx + 1) % len(active_nodes)

        ## Print summary of submitted jobs
        node_job_counts = list(node_job_counts.items())
        # Sort by nodename
        node_job_counts.sort(key=lambda x: x[0])
        node_jobs_info = tabulate(node_job_counts,
                                  headers=['Node', 'Num_jobs'],
                                  tablefmt="rounded_outline",
                                  colalign=("left", "right"))
        # Sort by nodename
        submitted_jobs.sort(key=lambda x: x[1])
        job_info = tabulate(
            submitted_jobs,
            headers=['Job Directory', 'Node', 'Job ID', 'Num_SMILES'],
            tablefmt="simple",
            colalign=("left", "left", "left", "right"))
        logger.info(
            f"Submitted {len(submitted_jobs)} jobs:\n{node_jobs_info}\n{job_info}"
        )
        print(
            f"Total {len(submitted_jobs)} jobs on {len(active_nodes)} nodes:\n{node_jobs_info}"
        )


def _submit_job(output_dir: str, node_name: str, cached_config: str,
                batch_file: str) -> str | None:
    """
    Launch a single sbatch job
    """
    # Get absolute paths for the batch input file (in pkl format)
    batch_file = Path(batch_file).resolve()
    batch_dir_name = batch_file.parent.name

    script_path = batch_file.parent / f"submit.sh"

    job_log_dir = output_dir / "slurm_logs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    job_log_path = job_log_dir / f"{batch_dir_name}.log"

    # Create the SLURM command
    slurm_cmd = SLURM_CMD.format(JOB_NAME=batch_dir_name.lstrip('fs-sn-'),
                                 JOB_LOG=job_log_path,
                                 NODE_NAME=node_name,
                                 PYTHON_EXE=_PYTHON_EXE,
                                 CACHED_CONFIG=cached_config,
                                 BATCH_SMILES_FILE=batch_file)

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
