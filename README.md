# Learning Calibratable Policies using Programmatic Style-Consistency [(arXiv)](https://arxiv.org/abs/1910.01179)

## Code

Code is written in Python 3.7.4 and [PyTorch](https://pytorch.org/) v.1.0.1. Will be updated for PyTorch 1.3 in the future.

## Usage

Train models with:

`$ python run_single.py -d <device id> --config_dir <config folder name>`

Not specifying a device will use CPU by default. See JSONs in `configs\` to see examples of config files.

### Test Run

`$ python run_single.py --config_dir test --test_code` should run without errors.

## Scripts

`$ python scripts/check_dynamics_loss.py -f <config folder name>` will compute and visualize the dynamics model error, where applicable.

`$ python scripts/compute_stylecon_ctvae.py -f <config folder name>` will compute the style-consistency.

`$ python scripts/visualize_samples_ctvae.py -f <config folder name>` will sample and save trajectories for each label class.
