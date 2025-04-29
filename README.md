# mqc_pipeline
High-throughput molecular quantum chemistry (MQC) pipeline

## Installation
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
