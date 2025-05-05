# mqc_pipeline
High-throughput molecular quantum chemistry (MQC) pipeline

## Mamba Installation
Dependencies are managaed by [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html),
which is a lightweight (and faster) alternative to conda. Installed it with:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
and add micromamba to the system path:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Clone this repository and navigate to the project directory:
```
git clone git@github.com:Solid-Energy-Systems/mqc_pipeline.git
cd mqc_pipeline
```

Use the provided script to create and manage the environment:
```
# Create a new environment
bash manage_env.sh create

# Update an existing environment
bash manage_env.sh update

# Clean the micromamba package cache
bash manage_env.sh clean
```
Most dependencies are specified in the `environment.yml` file.
Update `manage_env.sh` script to change the path of the enviroment,
which is set to `$HOME/mqc-env` by default.

Activate the environment:
```
micromamba activate ~/mqc-env
```

Install [aimnet2calc](https://github.com/zubatyuk/aimnet2calc) to use
AIMNet2 neural network potential for geometry optimization
```shell
bash install_aimnet2calc.sh
```

## Command-line API

The command-line interface is designed for processing large batches of molecules on SLURM-managed clusters.
The system automatically distributes molecules across nodes and GPUs, handles job submission,
and consolidates results into a single CSV file.

### Basic Usage

Given a CSV or TXT file with a single column of SMILES strings:
```
python scripts/slurm_runner.py --input <CSV_OR_TXT> --num-nodes <NUM_NODES> --gpu-per-node <GPU_PER_NODE>
```

Given a directory with multiple XYZ files:
```
python scripts/slurm_runner.py --xyz-dir <XYZ_DIR> --num-nodes <NUM_NODES> --gpu-per-node <GPU_PER_NODE>
```

### Available Options

| Option | Description |
|--------|-------------|
| `--input` | Path to a file containing SMILES strings (one per line) |
| `--xyz-dir` | Path to a directory containing XYZ files |
| `--num-nodes` | Number of nodes to distribute the workload across |
| `--gpu-per-node` | Number of GPUs per node |
| `--output-dir` | Directory to store output files (default: mqc_output) |
| `--final-csv` | Path to final consolidated CSV file (default: <output-dir>/final_results.csv) |
| `--partition` | SLURM partition to use (default: high-priority) |
| `--time-limit` | Time limit for each job (default: 24:00:00) |
| `--mem` | Memory limit for each job (default: 100G) |
| `--consolidate-only` | Only consolidate existing results without submitting new jobs |

### Examples

Process a file with 1000 SMILES on 2 nodes with 4 GPUs per node, saving to a custom output directory:
```
python scripts/slurm_runner.py --input compounds.csv --num-nodes 2 --gpu-per-node 4 --output-dir my_results
```

Process XYZ files using a specific SLURM partition with custom memory and time limits:
```
python scripts/slurm_runner.py --xyz-dir ./molecules --num-nodes 1 --gpu-per-node 8 --partition normal --mem 200G --time-limit 48:00:00
```

Consolidate previously processed results without submitting new jobs:
```
python scripts/slurm_runner.py --consolidate-only --output-dir previous_run --final-csv final_consolidated.csv
```

### Output Format

The system produces a consolidated CSV file with the following columns:

- `smiles` - SMILES string representation of the molecule
- `unique_id` - Unique identifier for the molecule
- `energy_hartree` - Energy in Hartree units
- `homo_eV` - HOMO orbital energy in eV
- `lumo_eV` - LUMO orbital energy in eV
- `esp_min` - Minimum electrostatic potential
- `esp_max` - Maximum electrostatic potential
- `dipole_x_debye` - X component of dipole moment in Debye
- `dipole_y_debye` - Y component of dipole moment in Debye
- `dipole_z_debye` - Z component of dipole moment in Debye

### How It Works

1. Load Balancing: The system distributes molecules evenly across nodes and GPUs.
1. SLURM Integration for HPC cluster processing: Each batch runs as a separate SLURM job on one GPU.
1. Thread-Safe Data Collection: Results from each job are safely collected with file locking.
1. Memory-Efficient Consolidation: Results are consolidated into a single CSV file using memory-efficient streaming.

