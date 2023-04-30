<h1 align="center">Score-based generative modeling</h1>


A minimal implementation of score-based generative modeling with
denoising score matching.

The implementation here is inspired by and partially based on
[https://github.com/tanelp/tiny-diffusion](https://github.com/tanelp/tiny-diffusion)
that  contains a minimal implementation of denoising diffusion
probabilistic models. While closely related, the code here is based on denoising score matching and we have used the following papers as a reference:



```bibtex
@inproceedings{
phillips2022spectral,
title={Spectral Diffusion Processes},
author={Angus Phillips and Thomas Seror and Michael John Hutchinson
    and Valentin De Bortoli and Arnaud Doucet and Emile Mathieu},
booktitle={NeurIPS 2022 Workshop on Score-Based Methods},
year={2022},
url={https://openreview.net/forum?id=bOmLb2i0W_h}
}

@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/alisiahkoohi/csgm
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
python scripts/train.py --dataset moons
```
## Questions

Please contact alisk@rice.edu for questions.

## Authors

Ali Siahkoohi and Lorenzo Baldassari