# mqc_pipeline
High-throughput molecular quantum chemistry (MQC) pipeline for geometry optimization and property calculation

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

The command-line interface, [mqc_runner.py](https://github.com/Solid-Energy-Systems/mqc_pipeline/blob/main/cmdline/mqc_runner.py)
is designed to automatically batch input molecules based on settings, handles SLURM job submission.

### Usage
1. Generate a default configuration file (good for starters)
```
python mqc_runner.py --write-default-config <config.yaml>
```
The content of the default config YAML looks like:
```yaml
# Path to a text/csv file that contains a single column of smiles strings. Alternatively, a directory containing multiple xyz files.
input_file_or_dir: 

# Number of SLURM batch jobs to launch. If set to 0, the pipeline will run locally without SLURM orchestration.
num_jobs: 1

# Name of the SLURM job. Only relevant when num_jobs > 0.
job_name: mqc_pipeline

# Method for geometry optimization.
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

# Additional DFT properties to compute. Supported properties: chelpg_charges, freq, quadrupole. Note that total electronic energy, HOMO/LUMO, dipole moment, ESP range are always returned.
additional_properties: ['chelpg_charges']

# Directory to save the output files. The default is the current working directory that cmdline unitlity is called.
output_dir: /mnt/filesystem/dev_renkeh/runner_test

# Output file format to write molecule-level and atom-level properties. Supported formats: csv, parquet. Default is csv.
output_file_format: csv

# Interval for logging progress during batch processing.
progress_log_interval: 10
```
Add input file path to the `input_file_or_dir` field. 

2. Run the pipeline with the configuration file:
```
python mqc_runner.py --config <config.yaml>
```

### Output Format

When using the default output format (`csv`), three files are generated:

---

#### 1. `molecule_property.csv`

Contains molecule-level properties for all molecules. Each row corresponds to a unique molecule and includes the following columns:

- **`unique_id`** — 11-digit UUID string uniquely identifying the molecule  
- **`smiles`** — SMILES string representation of the molecule  
- **`energy_hartree`** — Total energy in Hartree units  
- **`homo_eV`** — HOMO orbital energy (eV)  
- **`lumo_eV`** — LUMO orbital energy (eV)  
- **`dipole_x_debye`** — X component of dipole moment (Debye)  
- **`dipole_y_debye`** — Y component of dipole moment (Debye)  
- **`dipole_z_debye`** — Z component of dipole moment (Debye)  
- **`esp_min`** — Minimum electrostatic potential  
- **`esp_max`** — Maximum electrostatic potential  

---

#### 2. `atom_property.csv`

Contains per-atom properties for all molecules. Each row corresponds to an atom and includes:

- Element type  
- Cartesian coordinate 
- Force/gradient
- CHELPG partial charge
- **`unique_id`** — Matches the molecule's UUID from `molecule_property.csv`  
- **`smiles`** — SMILES string for the corresponding molecule  

The `unique_id` allows linking atoms to their parent molecules.

---

#### 3. `metadata.csv`

Stores performance-related information for each molecule. Columns include:

- **`unique_id`** — Molecule identifier  
- **`dft_opt_duration`** — Time spent on DFT optimization (seconds)  
- **`dft_prop_calc_duration`** — Time spent on DFT property calculation (seconds)  
- **`total_duration`** — Total computation time for the molecule (seconds)
