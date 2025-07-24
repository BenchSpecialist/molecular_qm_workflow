import os
import time
import json
import asyncio
import aiohttp
from typing import List, Optional
from pathlib import Path
from more_itertools import batched

from ase import Atoms
from ase.io.jsonio import MyEncoder, decode

from .common import Structure
from .util import get_default_logger

logger = get_default_logger()

URL = os.environ.get("INFERENCE_URL", 'http://localhost:8000/v1/infer')
DEFAULT_TIMEOUT = int(os.environ.get("TRITON_TIMEOUT", 600))  # seconds
MAX_RETRIES = int(os.environ.get("TRITON_MAX_RETRIES",
                                 5))  # number of retries for failed requests
RETRY_DELAY = int(os.environ.get("TRITON_RETRY_DELAY", 3))  # seconds

FAILED_INPUTS_FILE = "FAILED_INPUTS.txt"


async def request_async(atoms: list[Atoms],
                        url: str = URL,
                        timeout: int = DEFAULT_TIMEOUT,
                        max_retries: int = MAX_RETRIES,
                        retry_delay: int = RETRY_DELAY,
                        **params) -> list[Atoms]:
    """
    Send an asynchronous request to the Triton Inference Server with retry logic.

    :param atoms: List of ASE Atoms objects to optimize
    :param url: URL of the Triton Inference Server
    :param timeout: Timeout in seconds for the request
    :param max_retries: Maximum number of retries for failed requests
    :param retry_delay: Delay in seconds between retries
    :param params: Additional parameters to include in the request

    :return: List of optimized ASE Atoms objects
    """
    atoms_data = json.dumps(atoms, cls=MyEncoder)
    json_data = {
        "atoms": atoms_data,
    } | params
    headers = {'Content-Type': "application/json"}

    for attempt in range(max_retries + 1):
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(
                    timeout=client_timeout) as session:
                async with session.post(url, headers=headers,
                                        json=json_data) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    response_data = await response.json()
                    return decode(response_data['atoms'])
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries:
                # Use exponential backoff for retries
                wait_time = retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed (attempt {attempt+1}/{max_retries+1}). "
                    f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Request failed after {max_retries+1} attempts: {str(e)}")
                return []


async def optimize_sts_async(
        sts: list[Structure],
        batch_size: Optional[int] = None,
        timeout: int = DEFAULT_TIMEOUT) -> list[Structure]:
    """
    Optimize a list of structures using Triton Inference Server asynchronously.
    This function sends the structures to the server in batches if batch_size is specified.
    If batch_size is None, all structures are sent in a single request.

    :param sts: List of Structure objects to be optimized
    :param batch_size: Number of structures to send in each request. If None, all structures are sent at once
    :param timeout: Timeout in seconds for each request
    :return: List of optimized Structure objects
    """
    atoms_objs = [st.to_ase_atoms() for st in sts]
    # TODO: figure out how to set charge, multiplicity to the server via http requests
    params = {'charge': 0, 'mult': 1}
    t_start = time.perf_counter()

    if not batch_size:
        relaxed_atoms_objs = await request_async(atoms_objs,
                                                 URL,
                                                 timeout=timeout,
                                                 params=params)
    else:
        batches = list(batched(atoms_objs, batch_size))
        logger.info(
            f'TritonInference: {len(batches)} requests with number of molecules: '
            f'{[len(batch) for batch in batches]}')

        # Execute batches with semaphore to limit concurrency
        sem = asyncio.Semaphore(min(8,
                                    len(batches)))  # Limit concurrent requests

        async def run_batch(batch: List[Atoms]) -> List[Atoms]:
            async with sem:
                return await request_async(batch,
                                           URL,
                                           timeout=timeout,
                                           params=params)

        tasks = [run_batch(batch) for batch in batches]
        relaxed_batches = await asyncio.gather(*tasks)
        # Flatten the list of batches
        relaxed_atoms_objs = [
            atom for batch in relaxed_batches for atom in batch
        ]

    time_taken = time.perf_counter() - t_start
    logger.info(
        f'TritonInference: Relaxed {len(sts)} structures in {time_taken:.2f} seconds.'
    )
    opt_sts = [Structure.from_ase_atoms(atoms) for atoms in relaxed_atoms_objs]
    return opt_sts


def optimize_sts_by_triton(sts: list[Structure],
                           batch_size: Optional[int] = None,
                           timeout: int = DEFAULT_TIMEOUT) -> list[Structure]:
    """
    Synchronous wrapper for optimize_sts_async, used to call the asynchronous
    function in a synchronous context.

    :param sts: List of Structure objects to be optimized
    :param batch_size: Number of structures to send in each request. If None, all structures are sent at once
    :param timeout: Timeout in seconds for each request to the Triton server
    :return: List of optimized Structure objects with successful convergence
    """
    opt_sts = asyncio.run(optimize_sts_async(sts, batch_size, timeout))

    if failed_sts := [(index, st) for index, st in enumerate(opt_sts)
                      if st.metadata['triton_converged'] == False]:
        with open(FAILED_INPUTS_FILE, 'a') as f:
            f.write("\n".join([
                f"{st.smiles}: batch optimization failed to converge."
                for _, st in failed_sts
            ]))

        logger.warning(
            f'TritonInference: {len(failed_sts)} structures failed to converge'
        )

    # Remove un-converged structures from output
    converged_sts = [
        st for st in opt_sts if st.metadata['triton_converged'] == True
    ]
    return converged_sts
