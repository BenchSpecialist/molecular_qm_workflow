This repository provides tools for high-throughput molecular quantum chemical property calculations, supporting automated workflows for large sets of organic or inorganic molecules.
It enables users to compute properties such as *HOMO/LUMO energies, dipole moments,
electrostatic potential (ESP), ESP-fitted atomic charges, thermochemical properties,
and more* using DFT or machine learning interatomic potential.

The workflow can be run at scale using Docker containers, with separate images for CPU-only or GPU-accelerated versions.
It can also be executed on SLURM-based HPC clusters.
Input formats include SMILES, XYZ, or pickled structures; outputs are generated as CSV or Parquet files for easy analysis.

## Docker Usage

The workflow is available as containerized Docker images for both CPU-only and GPU-enabled execution.

### Build Docker Images

Two separate images are provided:

1. CPU-only image - Uses standard PySCF and PyTorch CPU
2. GPU-enabled image - Uses gpu4pyscf and CUDA-enabled PyTorch

```bash
docker build -f docker/Dockerfile.cpu -t mqc-pipeline:latest-cpu .
docker build -f docker/Dockerfile.gpu -t mqc-pipeline:latest-gpu .
```

### Run Containers

Run the CPU-only container with a volume mount to access your input files:
```bash
docker run --rm -v /path/to/local/workspace:/workspace mqc-pipeline:latest-cpu \
  python /app/docker/run.py --input /workspace/smiles.txt
```
For GPU support, use the `--gpus all` flag with the GPU-enabled image:
```bash
docker run --rm --gpus all -v /path/to/local/workspace:/workspace mqc-pipeline:latest-gpu \
  python /app/docker/run.py --input /workspace/smiles.txt
```

#### Customize calculation settings with configuration file
First, generate a default configuration file by running:
```bash
docker run --rm -v /path/to/local/workspace:/workspace mqc-pipeline:latest-cpu \
  python /app/docker/run.py --write-default-config /workspace/config.yaml
```
> See [example_config.yaml](docker/example_config.yaml) for all available options.

The `config.yaml` file will be created in your local workspace directory.
Edit this file to add your input file path to the `input_file_or_dir` field,
then run the container with the configuration file:
```bash
docker run --rm -v /path/to/local/workspace:/workspace mqc-pipeline:latest-cpu \
  python /app/docker/run.py --config /workspace/config.yaml
```

The container generates the following output files in the output directory (default: `output_dir`):
- `molecule_property.{csv|parquet}` - Molecular-level properties
- `atom_property.{csv|parquet}` - Atomic-level properties
- `metadata.csv` - Performance metrics
- `FAILED_INPUTS.txt` - Log of failed molecules (if any)

> See the [Output Schema](#output-schema) section for detailed documentation on the output data structure.

## Command-line interface

Two command-line entrypoints for different computational approaches are provided.
Both interfaces follow similar usage patterns with configuration YAML files and command-line arguments tailored to their respective computational backends.

### 1. `slurm_runner.py` - DFT-based workflow

The [`slurm_runner.py`](cmdline/slurm_runner.py) provides an end-to-end interface for high-throughput molecular geometry optimization via GPU-accelerated DFT or machine learning potential (AIMNet2) via ASE interface, and subsequent property calculations using DFT. It automatically:

- Batches input molecules based on configuration settings
- Launches batch jobs to the SLURM-managed GPU cluster
- Monitors and logs progress throughout execution
- Collects results from distributed jobs
- Combines output data from multiple batches into unified datasets
- Cleanup distributed job files after successful execution

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

To get started, first-time users should generate a default configuration file that includes all available options. This works just like the Docker example—run the following command to create a starter config file you can edit:

```bash
$ slurm_runner.py --write-default-config config.yaml
```

The configuration file includes all standard options listed in [example_config.yaml](docker/example_config.yaml), along with additional SLURM-specific fields.
```yaml
# Number of SLURM batch jobs to launch.
# If set to 0, the pipeline will run locally without SLURM orchestration.
num_jobs: 1

# Name of the SLURM job. Only relevant when num_jobs > 0.
job_name: test
```

Set the path to your input file using the `input_file_or_dir` field in the configuration file. After saving your changes, you can launch the workflow with:
```bash
$ slurm_runner.py --config config.yaml
```

### 2. `triton_runner.py` - ML-accelerated workflow

[`triton_runner.py`](cmdline/triton_runner.py) lets you run fast geometry optimization and property calculations using NVIDIA Triton inference servers and ML/DFT methods.

*Caveats: To use this workflow, you need to deploy an NVIDIA NIM container and set the `AIMNET_PROP_MODEL_PATH` environment variable to point to the location of the property model checkpoint file.*

Follow these steps strictly to run the Triton workflow:

1. Generate the `ACTIVE_TRITON_NODES.txt` file containing available Triton server nodes:
```bash
triton_runner.py --get-active-triton-nodes
```
This file is required in the config YAML.

2. Estimate your workload and request the total number of Triton server nodes to run:
```bash
triton_runner.py --request-num-server-nodes <num_active_nodes>
```
Please wait at least 2 minutes for all servers to initialize.

3. Generate a default configuration file:
```bash
triton_runner.py --write-default-config <config.yaml>
```
Edit this file to specify your input parameters and computational settings.

4. Run the workflow with the configuration file:
```
triton_runner.py --config <config.yaml>
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


