import numpy as np
import torch
import time
from pathlib import Path

from mqc_pipeline.common import Structure
from mqc_pipeline.constants import ELEMENT_TO_ATOMIC_NUMBER
from mqc_pipeline.util import get_default_logger

logger = get_default_logger()

from .aimnet2 import AIMNet2
from .config import load_yaml, build_module

AIMNET_PROP_MODEL_PT = Path(
    '/mnt/filesystem/dev_renkeh/aimnet-model-zoo/merged_model_181.pt')
AIMNET_MODEL_YAML = Path(__file__).parent / 'aimnet2.yaml'

_device_str = "cuda" if torch.cuda.is_available() else "cpu"
_device = torch.device(_device_str)

ESP_CLIP_RANGE = (-20.0, 20.0)
MO_CLIP_RANGE = (-20.0, 20.0)


def get_model():
    if AIMNET_MODEL_YAML.exists():
        config = load_yaml(config=AIMNET_MODEL_YAML)
        model = build_module(config)
        logger.info(f"Created AIMNet2 model from {AIMNET_MODEL_YAML}")
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
            f"YAML file not found, created AIMNet2 model with default parameters"
        )

    # Load state dict
    state_dict = torch.load(AIMNET_PROP_MODEL_PT,
                            map_location=_device,
                            weights_only=True)
    logger.info(
        f'Loaded model state dict from {AIMNET_PROP_MODEL_PT}, {len(state_dict)} parameters'
    )

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
    model = model.to(_device)
    return model


def sts_to_batch_input(sts: list[Structure]) -> dict[str, torch.Tensor]:
    """
    Convert a list of Structure objects to batch input for AIMNet2 model.
    Ideally, all input structures should have the same number of atoms,
    so there's no redundant data due to padding.

    :param sts: List of Structure objects to convert
    :return: Dictionary containing batched tensors ready for model input
        - coord: Coordinates tensor of shape (num_mols, n_atoms, 3)
        - numbers: Atomic numbers tensor of shape (num_mols, n_atoms)
        - charge: Molecular charges tensor of shape (num_mols,)
        - mult: Multiplicities tensor of shape (num_mols,)
    """
    # Get max number of atoms for padding
    n_atoms_distribution = [len(st.elements) for st in sts]
    if len(set(n_atoms_distribution)) > 1:
        logger.warning(
            "sts_to_batch_input: Input structures have varying number of atoms. Padding will be applied."
        )
        logger.info(f"Atom counts per molecule: {n_atoms_distribution}")
    n_atoms = max(len(st.elements) for st in sts)
    num_mols = len(sts)
    logger.info(
        f"sts_to_batch_input: Coordinates tensor shape: ({num_mols}, {n_atoms}, 3)"
    )

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
        "coord": torch.tensor(coords, dtype=torch.float32, device=_device),
        "numbers": torch.tensor(atomic_nums, dtype=torch.long, device=_device),
        "charge": torch.tensor(charges, dtype=torch.float32, device=_device),
        "mult": torch.tensor(mults, dtype=torch.float32, device=_device)
    }

    return batch_input


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
    other = ['energy', 'numbers', '_natom']
    if _device_str == "cuda":
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
        num_atoms = filtered_output['_natom'][i]
        assert st.atomic_numbers == atom_numbers[:num_atoms], \
            f'Atom number mismatched: {st.atomic_numbers} with {atom_numbers}'
        st.property['AIMNET2_energy_eV'] = filtered_output['energy'][i]
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


def run(
    sts: list[Structure] | Structure,
    mo_range: tuple[float, float] | None = None,
    esp_range: tuple[float, float] | None = None,
) -> list[Structure]:
    if isinstance(sts, Structure):
        sts = [sts]

    aimnet_model = get_model()
    batch_input = sts_to_batch_input(sts)

    t_start = time.perf_counter()

    with torch.no_grad():
        batch_output = aimnet_model(batch_input)

    if mo_range:
        for key in ['homo', 'lumo']:
            batch_output[key] = apply_clipping(batch_output[key], mo_range)
    if esp_range:
        for key in ['esp_min', 'esp_max']:
            batch_output[key] = apply_clipping(batch_output[key], esp_range)

    if _device_str == "cuda":
        torch.cuda.synchronize()

    add_properties_to_sts(batch_output, sts)

    logger.info(
        f"AIMNET property inference: {len(sts)} structures in {time.perf_counter() - t_start:.2f} seconds"
    )

    return sts
