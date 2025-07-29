import os
import time
import torch
import numpy as np
from pathlib import Path
from filelock import FileLock
from functools import lru_cache
from collections import namedtuple

from mqc_pipeline.common import Structure
from mqc_pipeline.constants import ELEMENT_TO_ATOMIC_NUMBER
from mqc_pipeline.util import get_default_logger

logger = get_default_logger()

from .aimnet2 import AIMNet2
from .config import load_yaml, build_module

# merged_model_181.pt contains 175 parameters
AIMNET_PROP_MODEL_PT = Path(
    '/mnt/filesystem/dev_renkeh/aimnet-model-zoo/merged_model_181.pt')
AIMNET_MODEL_YAML = Path(__file__).parent / 'aimnet2.yaml'

TOTAL_CPUS_PER_NODE = 128
# `torch.cuda.device_count()` gives 0 since when launching cmdline API on login node
AVAILABLE_GPUS = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))

ESP_CLIP_RANGE = (-20.0, 20.0)
MO_CLIP_RANGE = (-20.0, 20.0)

BatchInfo = namedtuple('BatchInfo', ['n_atoms_min', 'n_atoms_max'])

BATCH_INFO_FILE = Path("_prop_batch_info.csv")


def _log(msg: str, file: Path = BATCH_INFO_FILE) -> None:
    """
    Write message to file with process-safe locking.
    """
    with FileLock(f"{file}.lock"):
        with open(file, 'a') as fp:
            fp.write(f'{msg}\n')
            fp.flush()  # write content immediately


@lru_cache(maxsize=1)
def get_model_for_device(device_id: int = 0):
    """Get model for specific GPU with caching"""
    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Initialize model
    if AIMNET_MODEL_YAML.exists():
        config = load_yaml(config=AIMNET_MODEL_YAML)
        model = build_module(config)
    else:
        aev_params = {
            "rc_s": 5.0,
            "nshifts_s": 16,
        }

        model = AIMNet2(aev=aev_params,
                        nfeature=16,
                        d2features=True,
                        ncomb_v=12,
                        hidden=[[512, 380], [512, 380], [512, 380, 380]],
                        aim_size=256,
                        outputs={})
        logger.warning(
            f"{AIMNET_MODEL_YAML} not exist, created AIMNet2 model with default parameters"
        )

    # Get model weight
    try:
        state_dict = torch.load(AIMNET_PROP_MODEL_PT,
                                map_location=device,
                                weights_only=True)
    except Exception as e:
        try:
            state_dict = torch.load(AIMNET_PROP_MODEL_PT,
                                    map_location=device,
                                    weights_only=False)
            logger.debug(
                f'Loaded model weights(weights_only=False) from {AIMNET_PROP_MODEL_PT}, {len(state_dict)} parameters'
            )
        except Exception as e:
            err_msg = f"Failed to load model weights from {AIMNET_PROP_MODEL_PT}: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    # Load state dict into model
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.debug(
            f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )

    # Move model to device
    model = model.to(device)
    return model


def sts_to_batch_input(
        sts: list[Structure],
        device: torch.device) -> tuple[dict[str, torch.Tensor], BatchInfo]:
    """
    Convert a list of Structure objects to batch input for AIMNet2 model.
    Ideally, all input structures should have the same number of atoms,
    so there's no redundant data due to padding.

    :param sts: List of Structure objects to convert
    :param device: Device to which the tensors should be moved (e.g., 'cuda:0' or 'cpu')
    :return: Tuple of a dictionary containing batched tensors ready for model input
        - coord: Coordinates tensor of shape (num_mols, n_atoms, 3)
        - numbers: Atomic numbers tensor of shape (num_mols, n_atoms)
        - charge: Molecular charges tensor of shape (num_mols,)
        - mult: Multiplicities tensor of shape (num_mols,)
        and BatchInfo namedtuple with batch statistics
    """
    # Get max number of atoms for padding
    num_mols = len(sts)
    n_atoms_distribution = [len(st.elements) for st in sts]
    n_atoms = max(n_atoms_distribution)
    info = BatchInfo(n_atoms_min=min(n_atoms_distribution),
                     n_atoms_max=max(n_atoms_distribution))

    # Initialize arrays with padding
    coords = np.zeros((num_mols, n_atoms, 3))
    atomic_nums = np.zeros((num_mols, n_atoms), dtype=np.int64)
    charges = np.zeros(num_mols)
    mults = np.ones(num_mols)

    # Fill arrays with data
    for i, st in enumerate(sts):
        n_atoms_mol = len(st.elements)
        coords[i, :n_atoms_mol] = st.xyz

        # Convert elements to atomic numbers if needed
        if st.atomic_numbers is not None:
            atomic_nums[i, :n_atoms_mol] = st.atomic_numbers
        else:
            atomic_numbers_list = [
                ELEMENT_TO_ATOMIC_NUMBER[el] for el in st.elements
            ]
            atomic_nums[i, :n_atoms_mol] = atomic_numbers_list
            st.atomic_numbers = atomic_numbers_list

        charges[i] = st.charge
        mults[i] = st.multiplicity

    # Convert to PyTorch tensors
    batch_input = {
        "coord": torch.tensor(coords, dtype=torch.float32, device=device),
        "numbers": torch.tensor(atomic_nums, dtype=torch.long, device=device),
        "charge": torch.tensor(charges, dtype=torch.float32, device=device),
        "mult": torch.tensor(mults, dtype=torch.float32, device=device)
    }

    return batch_input, info


# Keys in batch_output: 'coord', 'numbers', 'charge', 'mult',
# '_nb_mode', 'mask_i', 'mask_ij', '_input_padded', '_natom', 'mol_sizes',
# 'mask_ij_lr', 'a', 'd_ij', 'gs', 'gv', '_delta_Q', '_dQ',
# 'charges', 'aim', 'aim_enhancedesp', 'aim_enhanced_electronic',
# 'energy', 'd_ij_lr', '_attention_weights',
# '_homo_atomic_weights', '_lumo_atomic_weights', '_orbital_mean', '_orbital_std',
# 'homo', 'lumo', 'homo_lumo_gap',
# '_esp_min_atomic_weights', '_esp_max_atomic_weights', '_esp_mean', '_esp_std',
# 'esp_min', 'esp_max', 'esp_gap'
def add_properties_to_sts(batch_output: dict[str, torch.Tensor],
                          sts: list[Structure]):

    properties = ['homo', 'lumo', 'esp_min', 'esp_max']
    other = ['energy', 'numbers', '_natom', 'charges']
    if torch.cuda.is_available():
        filtered_output = {
            k: v.cpu().numpy().tolist()
            for k, v in batch_output.items() if k in properties + other
        }
    else:
        filtered_output = {
            k: v.numpy().tolist()
            for k, v in batch_output.items() if k in properties + other
        }

    for i, st in enumerate(sts):
        atom_numbers = filtered_output['numbers'][i]
        num_atoms = filtered_output['_natom']
        if not isinstance(num_atoms, int):
            num_atoms = filtered_output['_natom'][i]
        assert st.atomic_numbers == atom_numbers[:num_atoms], \
            f'Atom number mismatched: {st.atomic_numbers} with {atom_numbers}'

        # This energy value should almost be equal to `st.property['triton_energy_ev']`,
        # if the Triton server and the extended Aimnet2 uses the same checkpoint
        st.metadata['AIMNET2_energy_eV'] = filtered_output['energy'][i]

        st.atom_property['AIMNET2_charges'] = filtered_output['charges'][
            i][:num_atoms]

        st.property.update({
            prop_key: filtered_output[prop_key][i][0]
            for prop_key in properties
        })


def apply_clipping(values: torch.Tensor, range: tuple[float,
                                                      float]) -> torch.Tensor:
    """
    Apply clipping to a tensor based on the specified range.

    Args:
        values: Input tensor to clip
        range: Tuple specifying the (min, max) clipping range

    Returns:
        Clipped tensor
    """
    # Clip input tensor in-place
    # values.clamp_(range[0], range[1])
    clipped_values = torch.clamp(values, range[0], range[1])
    n_clipped = torch.sum((values < range[0]) | (values > range[1])).item()
    if n_clipped > 0:
        logger.info(f"Clipped {n_clipped}/{len(values)} values to {range}")
    return clipped_values


def run_one_batch(
    device_id: int,
    sts: list[Structure],
    batch_id: int | None = None,
    mo_range: tuple[float, float] | None = None,
    esp_range: tuple[float, float] | None = None,
) -> list[Structure]:
    """Process a batch of structures on specified GPU"""

    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Get model for this device
    aimnet_model = get_model_for_device(device_id)

    # Convert structures to batch input
    batch_input, info = sts_to_batch_input(sts, device=device)

    # Run inference
    t_start = time.perf_counter()
    with torch.no_grad():
        batch_output = aimnet_model(batch_input)

    # Apply clipping if needed
    if mo_range:
        for key in ['homo', 'lumo']:
            batch_output[key] = apply_clipping(batch_output[key], mo_range)
    if esp_range:
        for key in ['esp_min', 'esp_max']:
            batch_output[key] = apply_clipping(batch_output[key], esp_range)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    add_properties_to_sts(batch_output, sts)

    _log(
        f'{batch_id},{str(device)},{len(sts)},{info.n_atoms_min},{info.n_atoms_max},{time.perf_counter() - t_start:.2f}'
    )

    return sts


def _batch(sts: list, size_range: tuple[int, int],
           num_cores: int) -> list[list]:
    """
    Divide a list of structures into batches based on available cores and size constraints.

    :param sts: List of structures to batch
    :param size_range: Tuple containing (min_batch_size, max_batch_size)
    :param num_cores: Number of CPU/GPU cores available for processing

    :return: List of batched structure lists
    """
    min_batch_size, max_batch_size = size_range

    # Calculate optimal batch size based on available cores
    if (batch_size :=
        (len(sts) + num_cores - 1) // num_cores) > max_batch_size:
        batch_size = max_batch_size
    elif batch_size < min_batch_size:
        batch_size = min_batch_size

    return [sts[i:i + batch_size] for i in range(0, len(sts), batch_size)]


def run_parallel(
        sts: list[Structure] | Structure,
        mo_range: tuple[float, float] | None = None,
        esp_range: tuple[float, float] | None = None,
        min_batch_size: int = 32,
        max_batch_size: int = 4096,
        num_cpu_workers: int = TOTAL_CPUS_PER_NODE) -> list[Structure]:
    """
    Run inference in parallel across multiple GPUs or CPUs.

    :param sts: List of Structure objects or a single Structure
    :param mo_range: Optional tuple specifying (min, max) clipping range for molecular orbital properties
    :param esp_range: Optional tuple specifying (min, max) clipping range for ESP properties
    :param max_batch_size: Batch size limit for structures
    :param num_cpu_workers: Number of CPU workers to use when no GPUs are available
    :return: List of structures with properties added
    """
    # Convert single structure to list if needed
    if isinstance(sts, Structure):
        sts = [sts]

    # Return early for empty input
    if not sts:
        return sts

    t_start = time.perf_counter()

    # Remove exiting BATCH_INFO_FILE
    if BATCH_INFO_FILE.exists():
        BATCH_INFO_FILE.unlink()
    # Add header to BATCH_INFO_FILE
    _log(msg='batch_id,device,num_mols,n_atoms_min,n_atoms_max,time')

    if torch.cuda.is_available():
        # GPU processing - split structures across available GPUs
        batches = _batch(sts,
                         size_range=(min_batch_size, max_batch_size),
                         num_cores=AVAILABLE_GPUS)

        logger.info(
            f"Parallel inference on {AVAILABLE_GPUS} GPUs with {len(batches)} batches"
        )

        # Use multiprocessing to run on multiple GPUs
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(processes=AVAILABLE_GPUS) as pool:
            results = pool.starmap(
                run_one_batch,
                [(i % AVAILABLE_GPUS, batch, i, mo_range, esp_range)
                 for i, batch in enumerate(batches)])
    else:
        # CPU processing - use multiple CPU workers
        num_cpu_workers = min(num_cpu_workers, len(sts))

        # Split structures across CPU workers
        batches = _batch(sts,
                         size_range=(min_batch_size, max_batch_size),
                         num_cores=num_cpu_workers)

        logger.info(
            f"Parallel inference on {num_cpu_workers} CPUs with {len(batches)} batches ({[len(batch) for batch in batches]})"
        )

        # Use multiprocessing to run on multiple CPU cores
        with torch.multiprocessing.Pool(processes=num_cpu_workers) as pool:
            results = pool.starmap(run_one_batch,
                                   [(0, batch, batch_id, mo_range, esp_range)
                                    for batch_id, batch in enumerate(batches)])

    # Flatten results
    out_sts = [st for batch in results for st in batch]

    # Verify all structures were processed
    assert len(out_sts) == len(sts), "Not all molecules were processed"

    logger.info(
        f"Property inference: {len(sts)} molecules in {time.perf_counter() - t_start:.2f} seconds"
    )

    # Remove .lock file after processing
    lock_file = Path(f'{BATCH_INFO_FILE.stem}.csv.lock')
    if lock_file.exists():
        lock_file.unlink()

    return out_sts
