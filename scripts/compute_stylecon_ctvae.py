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


def compute_stylecon_ctvae(exp_dir, trial_id, args):
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

    # Load batch
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    (states, actions, labels_dict) = next(iter(loader))
    states = states.transpose(0,1)
    actions = actions.transpose(0,1)
 
    # Randomly permute labels for independent sampling
    if args.sampling_mode == 'indep':
        for lf_name, labels in labels_dict.items():
            random_idx = torch.randperm(labels.size(0))
            labels_dict[lf_name] = labels[random_idx]

    labels_concat = torch.cat(list(labels_dict.values()), dim=-1) # MC sample of labels

    # Generate rollouts with labels
    with torch.no_grad():
        env.reset(init_state=states[0].clone())
        model.reset_policy(labels=labels_concat, temperature=args.temperature)
        
        rollout_states, rollout_actions = generate_rollout(env, model, burn_in=args.burn_in, burn_in_actions=actions, horizon=actions.size(0))
        rollout_states = rollout_states.transpose(0,1)
        rollout_actions = rollout_actions.transpose(0,1)

    stylecon_by_sample = torch.ones(args.num_samples) # used to track if ALL categorical labels are self-consistent
    categorical_lf_count = 0

    for lf in dataset.active_label_functions:
        print('--- {} ---'.format(lf.name))
        y = labels_dict[lf.name]

        # Apply labeling functions on rollouts
        rollouts_y = lf.label(rollout_states, rollout_actions, batch=True)

        if lf.categorical:
            # Compute stylecon for each label class
            matching_y = y*rollouts_y
            class_count = torch.sum(y, dim=0)
            stylecon_class_count = torch.sum(matching_y, dim=0)
            stylecon_by_class = stylecon_class_count/class_count
            stylecon_by_class = [round(i,4) for i in stylecon_by_class.tolist()]

            # Compute stylecon for each sample
            stylecon_by_sample *= torch.sum(matching_y, dim=1)
            categorical_lf_count += 1
            
            print('class_sc_cnt:\t {}'.format(stylecon_class_count.int().tolist()))
            print('class_cnt:\t {}'.format(class_count.int().tolist()))
            print('class_sc:\t {}'.format(stylecon_by_class))
            print('average: {}'.format(torch.sum(stylecon_class_count)/torch.sum(class_count)))

        else:
            # Compute stylecon
            diff = rollouts_y-y
            print('L1 stylecon {}'.format(torch.mean(torch.abs(diff)).item()))
            print('L2 stylecon {}'.format(torch.mean(diff**2).item()))

            # Visualizing stylecon
            range_lower = dataset.summary['label_functions'][lf.name]['train_dist']['min']
            range_upper = dataset.summary['label_functions'][lf.name]['train_dist']['max']

            label_values = np.linspace(range_lower, range_upper, args.num_values)
            rollouts_y_mean = np.zeros(args.num_values)
            rollouts_y_std = np.zeros(args.num_values)

            for i, val in enumerate(label_values):
                # Set labels
                # TODO this is not MC-sampling, need to do rejection sampling for true computation I think
                labels_dict_copy = { key: value for key, value in labels_dict.items() }
                labels_dict_copy[lf.name] = val*torch.ones(args.num_samples, 1)
                labels_concat = torch.cat(list(labels_dict_copy.values()), dim=-1)

                # Generate samples with labels
                with torch.no_grad():
                    samples = model.generate(x, labels_concat, burn_in=args.burn_in, temperature=args.temperature)
                    samples = samples.transpose(0,1)

                # Apply labeling functions on samples
                rollouts_y = lf.label(samples, batch=True)

                # Compute statistics of labels
                rollouts_y_mean[i] = torch.mean(rollouts_y).item()
                rollouts_y_std[i] = torch.std(rollouts_y).item()

            plt.plot(label_values, label_values, color='b', marker='o')
            plt.plot(label_values, rollouts_y_mean, color='r', marker='o')
            plt.fill_between(label_values, rollouts_y_mean-2*rollouts_y_std, rollouts_y_mean+2*rollouts_y_std, color='red', alpha=0.3)
            plt.xlabel('Input Label')
            plt.ylabel('Output Label')
            plt.title('LF_{}, {} samples, 2 stds'.format(lf.name, args.num_samples))
            plt.savefig(os.path.join(trial_dir, 'results', '{}.png'.format(lf.name)))
            plt.close()

    stylecon_all_count = int(torch.sum(stylecon_by_sample))
    print('--- stylecon for {} categorical LFs: {} [{}/{}] ---'.format(
        categorical_lf_count, stylecon_all_count/args.num_samples, stylecon_all_count, args.num_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--exp_folder', type=str,
                        required=True, default=None,
                        help='folder of experiments from which to load models')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('-n', '--num_samples', type=int,
                        required=False, default=200,
                        help='total number of samples')
    parser.add_argument('-v', '--num_values', type=int,
                        required=False, default=20,
                        help='number of values to evaluate for continuous LFs')
    parser.add_argument('-b', '--burn_in', type=int,
                        required=False, default=0,
                        help='burn in period, for sequential data')
    parser.add_argument('-t', '--temperature', type=float,
                        required=False, default=1.0,
                        help='sampling temperature')
    parser.add_argument('--sampling_mode', type=str,
                        required=False, default='mc', choices=['mc', 'indep'],
                        help='how to sample labels for computing self-consistency')
    args = parser.parse_args()

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    # Check self consistency
    for trial_id in master['summaries']:
        compute_stylecon_ctvae(exp_dir, trial_id, args)
    