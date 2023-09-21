# About
Repository implementing a protocol to generate structural ensembles of intrinsically disordered regions (IDRs) via the [idpGAN](https://github.com/feiglab/idpgan) and [cg2all](https://github.com/huhlim/cg2all) machine learning models. This is the same protocol used to generate the idpGAN ensembles deposited on the [Protein Ensemble Database](https://proteinensemble.org) (PED), see an example [here](https://proteinensemble.org/entries/PED00457).

# Install

## Premise
We recommend to install and run this package in a new [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that you create from the `idpgan_ped.yml` file in this repository.

## Installation instructions
1. Fetch this repository:
   ```bash
   git clone https://github.com/giacomo-janson/idpgan_ped.git
   ```
   and go into the root directory of the repository.
2. Install the dedicated conda enviroment:
   ```bash
   conda env create -f idpgan_ped.yml
   ```
3. Download [idpGAN](https://github.com/feiglab/idpgan): run the `scripts/download_idpgan.sh` script.
4. Install [cg2all](https://github.com/huhlim/cg2all) for converting C-alpha conformations to all atom. This will also install the right version of PyTorch, which is needed to run idpGAN:
   ```bash
   pip install git+http://github.com/huhlim/cg2all
   ```

# How to generate ensembles
1. Go into the root directory of this repository.
2. Prepare a JSON input file with the names and sequences of the IDRs that you would like to model. See the `data/input.template.json` file of this repository. You can edit it and use your own sequences.
3. Run the following command:
   ```bash
   python scripts/generate_ensembles.py -i input.json -c config/main.json
   ```
   it will actually generate the ensembles. The `-c` argument specifies a configuration file where you can edit the output directory. The default arguments are the ones used for the ensembles deposited on PED, but you can modify them according to your needs (e.g.: changing the number of conformations in an ensemble with `--n_samples`).