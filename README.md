<h1 align="center">Conditional score-based diffusion models for Bayesian inference in infinite dimensions</h1>

Code to reproduce the results in the paper [Conditional score-based diffusion models for Bayesian inference in infinite dimensions](https://arxiv.org/abs/2305.19147).

## Installation

Run the commands below to install the required packages.

```bash
cd csgm/
conda env create -f environment.yml
conda activate csgm
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate csgm`, the
following times.

## Usage

To run the example scripts, you can use the following commands.

```bash
python scripts/train_conditional_quadratic.py # Toy example
python scripts/train_conditional_seismic_imaging.py # Seismic imaging example
```

## Preliminary results

A summary of results for the toy quadratic and seismic imaging examples can be found [here](https://www.dropbox.com/scl/fi/7ynmvlxlhrb88kn9epuob/supp.pdf?rlkey=097c13aayrrs7ktahic8egq7q&dl=0).


## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi




