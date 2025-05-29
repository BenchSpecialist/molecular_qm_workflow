# mqc_pipeline
High-throughput molecular quantum chemistry (MQC) pipeline for geometry optimization and property calculation

## Setup

For detailed environment setup and installation instructions, please refer to [env_setup.md](env_setup.md).

### Pre-configured Environment

The complete runtime environment for this pipeline has already been configured on the **Fluidstack cluster**. To access the cluster, connect via SSH:

```bash
ssh slurm@195.242.10.37
```

## Command-line API

The command-line interface [`mqc_runner.py`](https://github.com/Solid-Energy-Systems/mqc_pipeline/blob/main/cmdline/mqc_runner.py) provides a end-to-end solution for high-throughput molecular quantum chemistry calculations. It automatically:

- **Batches** input molecules based on configuration settings
- **Launches** parallelized jobs to the SLURM-managed GPU cluster
- **Monitors** and logs progress throughout execution
- **Collects** results from distributed jobs
- **Cleanups** distributed job files after data collection

### Usage

The command-line utility has been pre-configured for the `slurm` user on the **Fluidstack cluster**. To get started, test the installation by running:
```
slurm@fs-s-login-001:~$ mqc_runner.py --help
usage: mqc_runner.py [-h] [--config CONFIG] [--write-default-config WRITE_DEFAULT_CONFIG] [--dry-run]
                     [--combine-results] [--cleanup]

Cmdline utility to run molecular geometry optimization and property calculation pipeline.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the YAML configuration file.
  --write-default-config WRITE_DEFAULT_CONFIG
                        Write the default configuration file and exit.
  --dry-run             For debug: Batch inputs, write sbatch script and print command without executing it.
  --combine-results     Combine results from all batches.
  --cleanup             Remove temporary files after combining results. Deletes batch directories, SLURM
                        scripts/logs, and cached config. Can only be used with --combine-results to prevent
                        accidental deletion of batch results before they are combined into final output files. Use
                        this to clean workspace after successful pipeline completion.
```


#### Step 1: Generate a Default Configuration File

For first-time users, generate a default configuration file with all available options:

```bash
slurm@fs-s-login-001:~$ mqc_runner.py --write-default-config config.yaml
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
job_name: mqc_pipeline

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

The minimal requirement to initialize the pipeline is specifying the input file path via the input_file_or_dir field.

#### Step 2: Execute the Pipeline

Once your configuration file is properly set up, run the pipeline:
```bash
slurm@fs-s-login-001:~$ mqc_runner.py --config config.yaml
```

### Output Schema

The pipeline generates structured output files containing molecular properties and computational metadata. When using the default CSV format, three primary files are created:

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
| **`dft_opt_duration`** | Time spent on DFT geometry optimization |
| **`dft_prop_duration`** | Total time spent on DFT property calculations |
| **`total_duration`** | Total computation time for the molecule |

For some computationally expensive additional properties, detailed timing breakdown is provided:

| Column | Description |
|--------|-------------|
| **`dft_gradient_duration`** | Time spent on DFT gradient calculation |
| **`dft_hessian_duration`** | Time spent on DFT Hessian and thermochemistry calculations |
| **`dft_vdw_volume_duration`** | Time spent on van der Waals volume estimation |
| **`dft_polarizability_duration`** | Time spent on polarizability tensor calculation |

> **Note:** These columns are only populated when the corresponding calculations are performed based on the `additional_properties` configuration setting.

When using XYZ file inputs, input source information is included:

| Column | Description |
|--------|-------------|
| **`from_xyz_file`** | Full path of the input XYZ file |


