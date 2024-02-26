<h1 align="center">Conditional score-based diffusion models for Bayesian inference in infinite dimensions</h1>

Code to reproduce the results in the paper [Conditional score-based diffusion models for Bayesian inference in infinite dimensions](https://papers.nips.cc/paper_files/paper/2023/hash/4c79c359b3c5f077c0b955f93cb0f53e-Abstract-Conference.html).

## Installation

Run the commands below to install the required packages. Make sure to adapt the `pytorch-cuda` version to your CUDA version in `environment.yml`.

```bash
cd csgm/
conda env create -f environment.yml
conda activate csgm
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate csgm`, the
following times.

## Data

### Training data

The training data for the toy example is generated on-the-fly and the seismic imaging example's data will be downloaded to `data/` directory upon running the associated script. 

### Pretrained model

The pretrained model can be downloaded with the following command. Note that for the visualization script to use this model, the default values in associated configuration json files must be used.

```bash
mkdir -p "data/checkpoints/imaging_dataset-seismic_batchsize-128_max_epochs-500_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4"
wget -O "data/checkpoints/imaging_dataset-seismic_batchsize-128_max_epochs-500_lr-0.002_lr_final-0.0005_nt-500_beta_schedule-linear_hidden_dim-32_modes-24_nlayers-4/checkpoint_300.pth" "https://www.dropbox.com/scl/fi/ejl7j1yx129y2rfp2eyqk/checkpoint_300.pth?rlkey=hcwqr3zfjsjw6w5oud9he2c9i&dl=0" --no-check-certificate
```

## Usage

To run the example scripts for training a new model, the following commands can be used. The list of command line arguments and their default values can be found in the configuration json files in `configs/`.

### Training

```bash
python scripts/train_conditional_quadratic.py # Toy example
python scripts/train_conditional_seismic_imaging.py # Seismic imaging example
```

### Inference

Setting the command line argument `--phase test` will perform posterior sampling for both examples, which also includes plotting results for the toy example. In order to plot the results for the seismic imaging example, after performing inference, run the above script with `--phase plot` command line argument.


## Preliminary results

A summary of results for the toy quadratic and seismic imaging examples can be found [here](https://www.dropbox.com/scl/fi/7ynmvlxlhrb88kn9epuob/supp.pdf?rlkey=097c13aayrrs7ktahic8egq7q&dl=0).


## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi




