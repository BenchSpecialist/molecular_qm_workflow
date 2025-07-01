#!/mnt/filesystem/dev_renkeh/mqc-env/bin/python
"""
Commandline API to run Triton-ML and Triton-DFT pipeline.

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
"""
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

SLURM_CMD = """#!/bin/bash
#SBATCH --job-name=triton_{NODE_ID}
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
    parser = argparse.ArgumentParser(
        description=
        "Cmdline utility to run molecular geometry optimization via Triton inference server."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--request-num-server-nodes",
        type=int,
        metavar="NUM_SERVER_NODES",
        help="Specify the desired number of Triton server nodes to run. "
        "If requested nodes > active nodes: starts additional servers to reach the target; "
        "If requested nodes < active nodes: uses all currently active nodes; "
        f"active nodes are tracked in: {_ACTIVE_NODES_FILE}.")

    group.add_argument("--config",
                       type=str,
                       help="Path to the YAML configuration file. ")

    parser.add_argument(
        "--get-active-triton-nodes",
        action="store_true",
        help=
        f"Get list of active Triton server nodes and write to {_ACTIVE_NODES_FILE}."
    )

    parser.add_argument("--write-default-config",
                        type=str,
                        help="Write the default configuration file and exit.")

    return parser.parse_args()


def _distribute_inputs(num_inputs, num_nodes) -> list[int]:
    """ Distribute inputs across nodes for batch processing."""
    base = num_inputs // num_nodes
    remainder = num_inputs % num_nodes
    batch_sizes = [
        base + 1 if i < remainder else base for i in range(num_nodes)
    ]
    return batch_sizes


def main():
    args = _parse_args()
    active_node_file = Path(_ACTIVE_NODES_FILE).resolve()

    from mqc_pipeline.workflow import triton_server_util
    if args.get_active_triton_nodes:
        active_nodes = triton_server_util.get_active_server_nodes()
        active_node_file.write_text("\n".join(active_nodes))
        print(f"Active Triton server nodes: {active_nodes}")
        print(f"Wrote {active_node_file}.")
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
            active_nodes = [
                node_name.strip() for node_name in
                active_node_file.read_text().strip().splitlines()
            ]

        from mqc_pipeline.workflow.io import read_smiles

        output_dir = Path.cwd()

        _cached_config_path = output_dir / "_config.pkl"
        # Caching config for batch jobs
        with open(_cached_config_path, "wb") as f:
            pickle.dump(settings, f)

        # TODO: this reads all SMILES strings into memory; may not be suitable
        # for large datasets
        smiles_list = read_smiles(settings.input_file)
        batch_sizes = _distribute_inputs(num_inputs=len(smiles_list),
                                         num_nodes=len(active_nodes))
        submitted_jobs = []
        start_idx = 0
        for index, batch_size in enumerate(batch_sizes):
            node_name = active_nodes[index]
            batch_smiles = smiles_list[start_idx:start_idx + batch_size]
            start_idx += batch_size

            batch_dir = output_dir / f"{node_name}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'{node_name}: {len(batch_smiles)} SMILES strings')

            batch_file = batch_dir / "input_smiles.pkl"
            with open(batch_file, 'wb') as fh:
                pickle.dump(batch_smiles, fh)

            with change_dir(batch_dir):
                job_id = _submit_job(output_dir=output_dir,
                                     node_name=node_name,
                                     cached_config=str(_cached_config_path),
                                     batch_file=str(batch_file))
            if job_id:
                submitted_jobs.append((node_name, job_id))

        job_info = tabulate(submitted_jobs,
                            headers=['Node', 'Job ID'],
                            tablefmt="simple",
                            colalign=("left", "left"))
        logger.info(f"Submitted jobs:\n{job_info}")


def _submit_job(output_dir: str, node_name: str, cached_config: str,
                batch_file: str) -> str | None:
    """
    Launch a single sbatch job
    """
    script_dir = output_dir / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{node_name}.sh"

    job_log_dir = output_dir / "slurm_logs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    job_log_path = job_log_dir / f"{node_name}.log"

    # Get absolute paths for the batch input file (in pkl format)
    batch_file = Path(batch_file).resolve()

    # Create the SLURM command
    slurm_cmd = SLURM_CMD.format(NODE_ID=node_name[-3:],
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
