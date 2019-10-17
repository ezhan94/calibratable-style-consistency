import argparse
import json
import os
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
sys.path.append(sys.path[0] + '/..')

from lib.models import get_model_class
from util.datasets import load_dataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def check_selfcon_tvaep_dm(exp_dir, trial_id):
    print('########## Trial {} ##########'.format(trial_id))

    # Get trial folder
    trial_dir = os.path.join(exp_dir, trial_id)
    assert os.path.isfile(os.path.join(trial_dir, 'summary.json'))

    # Load config
    with open(os.path.join(exp_dir, 'configs', '{}.json'.format(trial_id)), 'r') as f:
        config = json.load(f)
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # Load dataset
    dataset = load_dataset(data_config)
    dataset.train()
    loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)

    # Load best model
    state_dict = torch.load(os.path.join(trial_dir, 'best.pth'), map_location=lambda storage, loc: storage)
    model_class = get_model_class(model_config['name'].lower())
    model_config['label_functions'] = dataset.active_label_functions
    model = model_class(model_config)
    model.filter_and_load_state_dict(state_dict)

    if not hasattr(model, 'dynamics_model'):
        return

    errors = []

    for batch_idx, (states, actions, _) in enumerate(loader):
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        with torch.no_grad():
            state_change = model.propogate_forward(states[0], actions[0])

        diff = torch.abs(state_change - (states[1]-states[0]))
        errors.append(diff.view(-1))

    errors = torch.cat(errors).numpy()
    print('Mean: {}'.format(np.mean(errors)))
    print('Median {}'.format(np.median(errors)))
    hist_range = [0.02*i for i in range(30)]

    N = len(errors)
    plt.hist(errors, bins=hist_range, weights=np.ones(N)/N, alpha=1, edgecolor='b')
    plt.xlim((0.0, 0.6))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Absolute Error')
    plt.ylabel('Percentage')
    plt.title('Dynamics model error')
    plt.savefig(os.path.join(trial_dir, 'results', 'dynamics_error.png'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--exp_folder', type=str,
                        required=True, default=None,
                        help='folder of experiments from which to load models')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    args = parser.parse_args()

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    # Check self consistency
    for trial_id in master['summaries']:
        check_selfcon_tvaep_dm(exp_dir, trial_id)
        