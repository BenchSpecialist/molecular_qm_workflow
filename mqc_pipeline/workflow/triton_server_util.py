import os
import time
import subprocess
from pathlib import Path
from tabulate import tabulate
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from mqc_pipeline.util import get_default_logger

logger = get_default_logger()

IMAGE_NAME = "registry.gitlab.com/nvidia/modulus/molecularstudio:aimnet2"
CONTAINER_NAME = "triton_server"

_TRITON_ENV_FILE = "triton_env_vars.dot"
_TRITON_CONTAINER_ENVS = """
MAX_ATOM_BATCH_SIZE = 6000
SKIN            = 0.1
MAX_STEP_SIZE   = 0.1
MAX_NUM_STEPS   = 1000
FORCE_TOLERANCE     = 1e-05
D_ENERGY_TOLERANCE  = 1e-3
F_MAX_LIMIT         = 1e3
DT              = 0.05
DT_MAX          = 1.0
DT_MIN          = 1e-3
N_MIN_INC       = 20
F_INC           = 1.1
F_DEC           = 0.5
ALPHA           = 0.25
F_ALPHA         = 0.99
DOWNHILL_CHECK  = True
ENABLE_CPU      = False
"""

DOCKER_CMD = """
sudo docker run -d --rm -it --gpus all -p 8000:8000 --name {CONTAINER_NAME} -e {ENV_FILE} {IMAGE_NAME}
"""

# node [3, 64] are used for production runs
FS_NODE_IDS = list(range(3, 65))
FS_NODENAME_TEMPLATE = "fs-sn-{NODE_ID:03d}"
# All production nodes on Fluidstack cluster
FS_NODE_NAMES = [f"fs-sn-{node_id:03d}" for node_id in FS_NODE_IDS]


def get_nodename_from_id(node_id: int) -> str:
    """
    Convert a node ID to its corresponding node name on the Fluidstack cluster.
    """
    if not isinstance(node_id, int):
        raise TypeError("Node ID must be an integer.")
    if not (1 <= node_id <= 64):
        raise ValueError("Node ID must be an integer between 1 and 64.")

    return FS_NODENAME_TEMPLATE.format(NODE_ID=int(node_id))


def get_id_from_nodename(node_name: str) -> int:
    """
    Convert a node name to its corresponding node ID on the Fluidstack cluster.
    """
    if not node_name.startswith("fs-sn-"):
        raise ValueError("Node name must start with 'fs-sn-'.")

    return int(node_name.split("-")[-1])


def _cleanup_pending_jobs() -> None:
    """
    Clean up pending SLURM jobs after timeout.
    """
    try:
        username = os.environ.get("USER")
        # Get pending jobs related to docker cmd for the current user
        squeue_cmd = f"squeue -u {username} -n sudo -t PD"
        result = subprocess.run(squeue_cmd.split(),
                                capture_output=True,
                                text=True,
                                check=False)
        lines = result.stdout.strip().splitlines()[1:]
        # Extract the first column (JOBID) from each line, ensuring the line is not empty
        job_ids = [line.strip().split()[0] for line in lines if line.strip()]

        if job_ids:
            # Cancel all pending jobs found
            scancel_cmd = ["scancel"] + job_ids
            subprocess.run(scancel_cmd, capture_output=True, check=False)
            print(f"Cleaned pending jobs due to node timeout: {job_ids}")
    except (subprocess.SubprocessError, Exception) as e:
        logger.warning(f"Error cleaning up pending jobs: {str(e)}")


#####################################
##### Check Server Availability #####
#####################################
def _check_docker_ps(node: str, timeout: float = 1.0) -> tuple[str, bool]:
    """
    Check if Triton server is running on a specific node with a timeout.
    Uses a minimal resource request and timeout to avoid hanging.

    :param node: Node name to check
    :param timeout: Timeout for the subprocess call
    :return: Tuple of (node_name, is_active)
    """
    # Add minimal CPU request and a timeout to prevent hanging
    docker_cmd = f"srun --nodelist={node} --partition=high-priority --time=00:00:30 sudo docker ps"
    try:
        # Add timeout to subprocess call to prevent hanging
        result = subprocess.run(
            docker_cmd.split(),
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            timeout=timeout)
        return node, result.returncode == 0 and IMAGE_NAME in result.stdout
    except subprocess.TimeoutExpired:
        msg = f"Timeout when checking docker processes on {node}"
        logger.warning(msg)
        print(msg)
        return node, False


def get_active_server_nodes() -> list[str]:
    """
    Get list of nodes where Triton server is running by checking docker processes in parallel.

    Uses thread pool to speed up network calls across multiple nodes, with timeouts to prevent
    hanging when nodes are unavailable.

    Performance characteristics:
    - Without timeout handling: ~0.3 seconds compared to ~12.3 seconds for sequential checks
    - With timeout handling: approximately timeout value + 0.3 seconds

    :return: List of node names where Triton server is running
    """
    t_start = time.perf_counter()
    # Use thread pool to check nodes in parallel
    # Number of workers based on number of nodes
    max_workers = len(FS_NODE_NAMES)

    active_nodes: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_check_docker_ps, node) for node in FS_NODE_NAMES
        ]
        for future in as_completed(futures):
            node, is_active = future.result()
            if is_active:
                active_nodes.append(node)

    _cleanup_pending_jobs()
    print(
        f"Checked {len(FS_NODE_NAMES)} nodes in {time.perf_counter() - t_start:.2f} seconds."
    )
    # Sort active nodes so nodes with the largest node IDs come first
    active_nodes.sort(reverse=True)
    return active_nodes


#########################
##### Stop Servers ######
#########################
def _stop_single_server(node_name: str) -> tuple[str, float, bool]:
    """
    Stop Triton server on a single node.

    :param node_name: Name of the node to stop server on
    :return: Tuple of (node_name, execution_time, success)
    """
    t_start = time.perf_counter()
    docker_cmd = f"srun --nodelist={node_name} sudo docker stop {CONTAINER_NAME}"

    try:
        subprocess.run(docker_cmd.split(),
                       capture_output=True,
                       text=True,
                       check=True)
        exec_time = time.perf_counter() - t_start
        return node_name, exec_time, True
    except subprocess.CalledProcessError as e:
        exec_time = time.perf_counter() - t_start
        logger.error(f"Failed to stop Triton server on {node_name}: {str(e)}")
        return node_name, exec_time, False


def stop_server_on_nodes(node_names: list[int]) -> list[str]:
    """
    Stop the Triton server on the specified nodes in parallel.

    :param node_ids: List of node IDs to stop servers on
    """
    # Use thread pool to stop servers in parallel
    max_workers = min(20, len(node_names))

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_stop_single_server, node_name=node_name)
            for node_name in node_names
        ]
        stopped_nodes = []
        for future in as_completed(futures):
            node, exec_time, success = future.result()
            if success:
                stopped_nodes.append(node)
                logger.debug(
                    f"Stopped Triton server on {node} in {exec_time:.2f} seconds."
                )

    logger.info(
        f"Stopped Triton server on {len(stopped_nodes)} nodes in {time.perf_counter() - t_start:.2f} seconds."
    )
    return stopped_nodes


#########################
##### Start Servers #####
#########################
@dataclass
class ServerStartResult:
    """Container for server start operation results"""
    node: str
    container_id: str
    success: bool
    error_msg: str = ""


def _start_single_server(node_name: str, base_cmd: list[str],
                         docker_cmds: list[str]) -> ServerStartResult:
    """Start Triton server on a single node.

    :param node: Node name (e.g., "fs-sn-003") to start server on
    :param base_cmd: Base SLURM command
    :param docker_cmds: Docker command to start the server
    :return: ServerStartResult containing operation results
    """
    cmd = base_cmd + [f'--nodelist={node_name}'] + docker_cmds

    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        container_id = result.stdout.strip().split()[-1]
        return ServerStartResult(node=node_name,
                                 container_id=container_id,
                                 success=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to launch Triton server on {node_name}: {str(e)}"
        logger.error(error_msg)
        return ServerStartResult(node=node_name,
                                 container_id="",
                                 success=False,
                                 error_msg=error_msg)


def start_server_on_nodes(node_names: list[str]) -> None:
    """Start the Triton server on the specified nodes in parallel.

    :param node_names: List of node names to start servers on
    """
    # Prepare environment file
    env_file = Path(_TRITON_ENV_FILE).resolve()
    env_file.write_text(_TRITON_CONTAINER_ENVS.strip())

    docker_cmd = DOCKER_CMD.format(CONTAINER_NAME=CONTAINER_NAME,
                                   ENV_FILE=str(env_file),
                                   IMAGE_NAME=IMAGE_NAME)

    # Prepare base SLURM command
    base_cmd = [
        "srun",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=128",  # Request all 128 CPU cores on each compute node
        "--gres=gpu:8",  # Request all 8 GPUs on each compute node
        "--partition=high-priority",
        "--mem=800G",
    ]

    # Use thread pool to start servers in parallel
    max_workers = min(20, len(node_names))
    successful_starts: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(_start_single_server,
                            node_name=node_name,
                            base_cmd=base_cmd,
                            docker_cmds=docker_cmd.split())
            for node_name in node_names
        ]

        for future in as_completed(futures):
            result = future.result()
            if result.success:
                successful_starts.append((result.node, result.container_id))

    # Generate report
    if successful_starts:
        node_container_info = tabulate(successful_starts,
                                       headers=['Node', 'Container ID'],
                                       tablefmt="simple",
                                       colalign=("left", "left"))
        msg = f"Started Triton server on nodes:\n{node_container_info}"
        logger.info(msg)
        print(msg)
        print(
            'IMPORTANT: Please wait at least 2 minutes for servers to start before sending http requests.'
        )
    else:
        msg = f"Failed to start any Triton servers on nodes: {node_names}"
        logger.error(msg)
        print(msg)

    # Clean up environment file
    env_file.unlink(missing_ok=True)
