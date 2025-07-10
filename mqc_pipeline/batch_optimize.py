import os
import time
import json
import asyncio
import aiohttp
from more_itertools import batched

from ase import Atoms
from ase.io.jsonio import MyEncoder, decode

from .common import Structure
from .util import get_default_logger

logger = get_default_logger()

URL = os.environ.get("INFERENCE_URL", 'http://localhost:8000/v1/infer')

FAILED_INPUTS_FILE = "FAILED_INPUTS.txt"


async def request_async(atoms: list[Atoms],
                        url: str = URL,
                        **params) -> list[Atoms]:
    """
    Send an asynchronous request to the Triton Inference Server.

    :param atoms: List of ASE Atoms objects to optimize
    :param url: URL of the Triton Inference Server
    :param params: Additional parameters to include in the request
    :return: List of optimized ASE Atoms objects
    """
    atoms_data = json.dumps(atoms, cls=MyEncoder)
    json_data = {
        "atoms": atoms_data,
    } | params
    headers = {'Content-Type': "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers,
                                json=json_data) as response:
            if response.status != 200:
                response.raise_for_status()
            response_data = await response.json()
            return decode(response_data['atoms'])


async def optimize_sts_async(sts: list[Structure],
                             batch_size: int | None = None) -> list[Structure]:
    """
    Optimize a list of structures using Triton Inference Server asynchronously.
    This function sends the structures to the server in batches if batch_size is specified.
    If batch_size is None, all structures are sent in a single request.

    :param sts: List of Structure objects to be optimized
    :param batch_size: Number of structures to send in each request. If None, all structures are sent at once
    :return: List of optimized Structure objects
    """
    atoms_objs = [st.to_ase_atoms() for st in sts]
    # TODO: figure out how to set charge, multiplicity to the server via http requests
    params = {'charge': 0, 'mult': 1}
    t_start = time.perf_counter()

    if not batch_size:
        relaxed_atoms_objs = await request_async(atoms_objs,
                                                 URL,
                                                 params=params)
    else:
        batches = list(batched(atoms_objs, batch_size))
        logger.info(
            f'TritonInference: {len(batches)} requests with number of molecules: '
            f'{[len(batch) for batch in batches]}')
        tasks = [request_async(batch, URL, params=params) for batch in batches]
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
                           batch_size: int | None = None) -> list[Structure]:
    """
    Synchronous wrapper for optimize_sts_async, used to call the asynchronous
    function in a synchronous context.
    """
    opt_sts = asyncio.run(optimize_sts_async(sts, batch_size))

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
