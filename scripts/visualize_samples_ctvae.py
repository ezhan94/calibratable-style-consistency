import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
sys.path.append(sys.path[0] + '/..')

from lib.models import get_model_class
from util.datasets import load_dataset
from util.environments import load_environment, generate_rollout

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def visualize_samples_ctvae(exp_dir, trial_id, num_samples, num_values, repeat_index, burn_in, temperature):
    print('#################### Trial {} ####################'.format(trial_id))

    # Get trial folder
    trial_dir = os.path.join(exp_dir, trial_id)
    assert os.path.isfile(os.path.join(trial_dir, 'summary.json'))

    # Load config
    with open(os.path.join(exp_dir, 'configs', '{}.json'.format(trial_id)), 'r') as f:
        config = json.load(f)
    data_config = config['data_config']
    model_config = config['model_config']

    # Load dataset
    dataset = load_dataset(data_config)
    dataset.eval()

    # Load best model
    state_dict = torch.load(os.path.join(trial_dir, 'best.pth'), map_location=lambda storage, loc: storage)
    model_class = get_model_class(model_config['name'].lower())
    assert model_class.requires_labels
    model_config['label_functions'] = dataset.active_label_functions
    model = model_class(model_config)
    model.filter_and_load_state_dict(state_dict)

    # Load environment
    env = load_environment(data_config['name']) # TODO make env_config?

    # TODO for now, assume just one active label function
    assert len(dataset.active_label_functions) == 1 

    for lf in dataset.active_label_functions:
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
        (states, actions, labels_dict) = next(iter(loader))

        if repeat_index >= 0:
            states_single = states[repeat_index].unsqueeze(0)
            states = states_single.repeat(num_samples,1,1)

            actions_single = actions[repeat_index].unsqueeze(0)
            actions = actions_single.repeat(num_samples,1,1)

        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        if lf.categorical:
            label_values = np.arange(0, lf.output_dim)
        else:
            range_lower = torch.min(dataset.lf_labels[lf.name])
            range_upper = torch.max(dataset.lf_labels[lf.name])

            label_values = np.linspace(range_lower, range_upper, num_values+2)
            label_values = np.around(label_values, decimals=1)
            label_values = label_values[1:-1]

        for c in label_values:
            if lf.categorical:
                y = torch.zeros(num_samples, lf.output_dim)
                y[:,c] = 1
            else:
                y = c*torch.ones(num_samples, 1)

            # Generate rollouts with labels
            with torch.no_grad():
                env.reset(init_state=states[0].clone())
                model.reset_policy(labels=y, temperature=args.temperature)
                
                rollout_states, rollout_actions = generate_rollout(env, model, burn_in=args.burn_in, burn_in_actions=actions, horizon=actions.size(0))
                rollout_states = rollout_states.transpose(0,1)
                rollout_actions = rollout_actions.transpose(0,1)

            dataset.save(
                rollout_states,
                rollout_actions,
                labels=y,
                lf_list=dataset.active_label_functions,
                burn_in=burn_in, 
                # save_path=os.path.join(trial_dir, 'results', '{}_label_{}'.format(lf.name, c)),
                # save_name='repeat_{:03d}'.format(repeat_index) if repeat_index >= 0 else '',
                save_path=os.path.join(trial_dir, 'results', lf.name),
                save_name='repeat_{:03d}_{}'.format(repeat_index, c) if repeat_index >= 0 else '',
                single_plot=(repeat_index >= 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--exp_folder', type=str,
                        required=True, default=None,
                        help='folder of experiments from which to load models')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('-n', '--num_samples', type=int,
                        required=False, default=8,
                        help='number of samples to generate FOR EACH CLASS')
    parser.add_argument('-v', '--num_values', type=int,
                        required=False, default=3,
                        help='number of values to evaluate for continuous LFs')
    parser.add_argument('-r', '--repeat_index', type=int,
                        required=False, default=-1,
                        help='repeated sampling with same burn-in')
    parser.add_argument('-b', '--burn_in', type=int,
                        required=False, default=0,
                        help='burn in period, for sequential data')
    parser.add_argument('-t', '--temperature', type=float,
                        required=False, default=1.0,
                        help='sampling temperature')
    args = parser.parse_args()

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    print(exp_dir)
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    assert args.repeat_index < args.num_samples
    if args.repeat_index >= 0:
        assert args.burn_in > 0

    # Check self consistency
    for trial_id in master['summaries']:
        visualize_samples_ctvae(exp_dir, trial_id, args.num_samples, args.num_values, args.repeat_index, args.burn_in, args.temperature)
    