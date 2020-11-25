# Learning Calibratable Policies using Programmatic Style-Consistency [(arXiv)](https://arxiv.org/abs/1910.01179)

## Demo

The demo will be live during ICML 2020 [here](http://basketball-ai.com/).

## Code

Code is written in Python 3.7.4 and [PyTorch](https://pytorch.org/) v.1.0.1. Will be updated for PyTorch 1.3 in the future.

## Usage

Train models with:

`$ python run_single.py -d <device id> --config_dir <config folder name>`

Not specifying a device will use CPU by default. See JSONs in `configs\` to see examples of config files.

### Test Run

`$ python run_single.py --config_dir test --test_code` should run without errors.

### Data

**[Update 11/25/20]** The basketball dataset is now available on [AWS Data Exchange](https://aws.amazon.com/marketplace/pp/prodview-7kigo63d3iln2?qid=1606330770194&sr=0-1&ref_=srh_res_product_title#offers). Please make sure to acknowledge Stats Perform if you use the data for your research. <br>

Download the basketball data into `util/datasets/bball/data/` (currently contains mock data).

To use your own data, you will need to create a new dataset in `util/datasets/` and create a new config folder in `configs/`.

## Scripts

`$ python scripts/check_dynamics_loss.py -f <config folder name>` will compute and visualize the dynamics model error, where applicable.

`$ python scripts/compute_stylecon_ctvae.py -f <config folder name>` will compute the style-consistency.

`$ python scripts/visualize_samples_ctvae.py -f <config folder name>` will sample and save trajectories for each label class.
