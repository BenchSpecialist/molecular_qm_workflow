This repository provides tools to run high-throughput molecular quantum chemical property calculations.

## Command-line interface

Two command-line entrypoints for different computational approaches are provided.
Both interfaces follow similar usage patterns with configuration YAML files and command-line arguments tailored to their respective computational backends.

### 1. `slurm_runner.py` - DFT-based workflow

The [`slurm_runner.py`](cmdline/slurm_runner.py) interface provides an end-to-end solution for high-throughput molecular geometry optimization via GPU-accelerated DFT or machine learning potential (AIMNet2) via ASE interface, and subsequent property calculations using DFT. It automatically:

- **Batches** input molecules based on configuration settings
- **Launches** parallelized jobs to the SLURM-managed GPU cluster
- **Monitors** and logs progress throughout execution
- **Collects** results from distributed jobs
- **Combines** output data from multiple batches into unified datasets
- **Cleanup** distributed job files after successful execution

### 2. `triton_runner.py` - ML-accelerated workflow

The [`triton_runner.py`](cmdline/triton_runner.py) interface provides access to fast geometry relaxations via NVIDIA Triton inference servers, and efficient property calculations via our internal ML models or DFT calculations. The cmdline utility provides functionality to:

- **Detect** or **launch** Triton inference server nodes
- **Distribute** molecular calculations across available server nodes
- **Process** results from all server nodes

### How to run Triton workflow

Please follow these steps strictly to run the Triton workflow:

1. Generate the `ACTIVE_TRITON_NODES.txt` file containing available Triton server nodes:
```bash
triton_runner.py --get-active-triton-nodes
```
This file is required in the config YAML.

2. Estimate your workload and request the total number of Triton server nodes to run:
```bash
triton_runner.py --request-num-server-nodes <num_active_nodes>
```
> **Important**: Please wait at least 2 minutes for all servers to initialize.

3. Generate a default configuration file:
```bash
triton_runner.py --write-default-config <config.yaml>
```
Edit this file to specify your input parameters and computational settings.

4. Run the workflow with the configuration file:
```
triton_runner.py --config <config.yaml>
```


### How to run DFT workflow

To get started, test the installation of the command-line utility by running:
```
$ slurm_runner.py --help
usage: slurm_runner.py [-h] [--config YAML_FILE] [--write-default-config YAML_FILE] [--dry-run] [--combine-results] [--cleanup] [--extract-xyz XYZ_DIR]

Cmdline utility to run molecular geometry optimization and property calculation pipeline.

options:
  -h, --help            show this help message and exit
  --config YAML_FILE    Path to the YAML configuration file.
  --write-default-config YAML_FILE
                        Write the default configuration file and exit.
  --dry-run             For debug: Batch inputs, write sbatch script and print command without executing it.
  --combine-results     Combine results from all batches.
  --cleanup             Remove temporary files after combining results. Deletes batch directories, SLURM scripts/logs, and cached config. Use this to clean workspace
                        after successful pipeline completion. Batch directories are only removed when this flag is used with --combine-results to prevent accidental
                        deletion of batch results before they are combined into final output files.
  --extract-xyz XYZ_DIR
                        Specify the directory and extract XYZ files from atom_property.csv.
```

#### Step 1: Generate a Default Configuration File

For first-time users, generate a default configuration file with all available options:

```bash
$ slurm_runner.py --write-default-config config.yaml
```

The generated configuration file contains the following structure and default values:
```yaml
# Path to a text/csv file that contains a single column of smiles strings.
# Alternatively, a directory containing multiple xyz files.
input_file_or_dir:

# Number of SLURM batch jobs to launch.
# If set to 0, the pipeline will run locally without SLURM orchestration.
num_jobs: 1

# Name of the SLURM job. Only relevant when num_jobs > 0.
job_name: test

# Method for geometry optimization.
# If set to `null` and use xyz inputs, geometry optimization is skipped.
# Supported methods: AIMNET2, DFT.
geometry_opt_method: DFT

# Name of the optimizer to use in ASE. Supported values: BFGS, FIRE
ase_optimizer_name: BFGS

# Convergence threshold for forces in ASE
ase_force_conv: 1e-05

# Maximum number of optimization cycles allowed in ASE
ase_max_cycle: 1000

# Basis set for PySCF calculations
pyscf_basis: 6311g*

# DFT functional for PySCF calculations
pyscf_functional: b3lypg

# Maximum number of SCF iterations allowed
pyscf_max_scf_cycle: 100

# SCF convergence tolerance
pyscf_scf_conv_tol: 1e-09

# Level of grid refinement for numerical integration
pyscf_grids_level: 3

# Whether to save the Fock matrix
pyscf_save_fock: False

# Solvent accessible region for ESP calculations in angstrom
esp_solvent_accessible_region: 3.0

# Grid spacing for ESP calculations in angstrom
esp_grid_spacing: 0.5

# Probe depth for ESP calculations in angstrom
esp_probe_depth: 1.1

# Additional DFT properties to compute.
# Supported properties: combustion_heat, chelpg_charges, forces, freq, vdw_volume, isotropic_polarizability, quadrupole
# Total electronic energy, HOMO/LUMO, dipole moment, ESP range are always returned.
additional_properties: ['combustion_heat', 'forces', 'chelpg_charges']

# Directory to save the output files.
# Default to the current working directory where the cmdline unitlity is called.
output_dir: /home/slurm

# Output file format to write molecule-level and atom-level properties.
# Supported formats: csv, parquet.
output_file_format: csv

# Interval for logging progress during batch processing.
progress_log_interval: 10
```

The minimal requirement for config is specifying the input file path via the input_file_or_dir field.

Once your configuration file is properly set up, start the workflow by:
```bash
$ slurm_runner.py --config config.yaml
```

### Output Schema

The workflow generates structured output files containing molecular properties and computational metadata. When using the default CSV format, three primary files are created:

---

#### 1. `molecule_property.csv` — Molecular-Level Properties

Contains molecule-level properties for all molecules. Each row corresponds to a unique molecule and by default includes the following columns:

| Column | Description |
|--------|-------------|
| **`unique_id`** | 16-digit UUID string uniquely identifying the molecule |
| **`smiles`** | SMILES string representation of the molecule |
| **`energy_hartree`** | Total electronic energy |
| **`homo_eV`** | Highest Occupied Molecular Orbital energy |
| **`lumo_eV`** | Lowest Unoccupied Molecular Orbital energy |
| **`dipole_x_debye`** | X component of dipole moment |
| **`dipole_y_debye`** | Y component of dipole moment |
| **`dipole_z_debye`** | Z component of dipole moment |
| **`esp_min`** | Minimum electrostatic potential |
| **`esp_max`** | Maximum electrostatic potential |

Additional columns are available based on the `additional_properties` field setting.

---

#### 2. `atom_property.csv` — Atomic-Level Properties

Contains detailed per-atom properties for all processed molecules. Each row corresponds to an individual atom and includes:

- **Element type** — Atomic symbol (e.g., C, H, O, N)
- **Cartesian coordinates** — 3D position (x, y, z)
- **Forces/gradients** — Atomic forces from optimization
- **CHELPG partial charges** — Electrostatic-derived atomic charges
- **`unique_id`** — Links to parent molecule in `molecule_property.csv`
- **`smiles`** — SMILES string for the corresponding molecule

> **Note:** Use the `unique_id` field to correlate atoms with their parent molecules across files.

---

#### 3. `metadata.csv` — Performance Metrics

Stores performance-related information for each molecule. It helps track computation times and identify performance bottlenecks.

By default this file contains:

| Column | Description |
|--------|-------------|
| **`unique_id`** | Molecule identifier (matches other output files) |
| **`dft_opt_time`** | Time spent on DFT geometry optimization |
| **`dft_prop_total_time`** | Total time spent on DFT property calculations |
| **`total_time`** | Total computation time for the molecule |

For some computationally expensive additional properties, detailed timing breakdown is provided:

| Column | Description |
|--------|-------------|
| **`dft_esp_time`**| Time spent on electrostatic potential calculation |
| **`dft_chelpg_time`**| Time spent on CHELPG charge calculation |
| **`dft_gradient_time`** | Time spent on DFT gradient calculation |
| **`dft_hessian_time`** | Time spent on DFT Hessian and thermochemistry calculations |
| **`dft_vdw_volume_time`** | Time spent on van der Waals volume estimation |
| **`dft_polarizability_time`** | Time spent on polarizability tensor calculation |

> **Note:** These columns are only populated when the corresponding calculations are performed based on the `additional_properties` configuration setting.

When using XYZ file inputs, input source information is included:

| Column | Description |
|--------|-------------|
| **`from_xyz_file`** | Full path of the input XYZ file |


