import argparse
import json
import os
import torch

from time import gmtime, strftime
from train import start_training

import subprocess
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, Manager

torch.backends.cudnn.benchmark = True

if mp.cpu_count() >= 32: 
    # should only be on my (albert) machine
    cpu_list = ','.join([str(x) for x in list(range(8)) + list(range(16, 24))])
    os.system("taskset -p -c {} {}".format(cpu_list, os.getpid()))

def run_config(exp_args):
    config_file = exp_args[0]
    config_dir = exp_args[1]
    save_dir = exp_args[2]
    exp_name = exp_args[3]
    device = exp_args[4]
    args = exp_args[5]
    master = exp_args[6]
    
    # Load JSON config file
    with open(os.path.join(config_dir, config_file), 'r') as f:
        config = json.load(f)

    trial_id = config_file[:-5] # remove .json at the end
    print('########## Trial {}:{} ##########'.format(exp_name, trial_id))

    # Create save folder
    save_path = os.path.join(save_dir, trial_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'checkpoints')) # for model checkpoints
        os.makedirs(os.path.join(save_path, 'results')) # for saving various results afterwards (e.g. plots, samples, etc.)

   # Start training
    summary, log, data_config, model_config, train_config = start_training(
        save_path=save_path,
        data_config=config['data_config'],
        model_config=config['model_config'],
        train_config=config['train_config'],
        device=device
    )

    # Save config file (for reproducability)
    config['data_config'] = data_config
    config['model_config'] = model_config
    config['train_config'] = train_config
    with open(os.path.join(save_dir, 'configs', config_file), 'w') as f:
        json.dump(config, f, indent=4)

    # Save summary file
    with open(os.path.join(save_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Save log file
    with open(os.path.join(save_path, 'log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    # Save entry in master file
    summary['log_path'] = os.path.join(args.save_dir, exp_name, trial_id, 'log.json')
    tmp = master['summaries'].copy()
    tmp[trial_id] = summary
    master['summaries'] = tmp

    # Save master file
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master._getvalue(), f, indent=4)



if __name__ == "__main__":
    mp.set_start_method('spawn')
    manager = Manager()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str,
                        required=False, default='',
                        help='path to all config files for experiments')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('--exp_name', type=str,
                        required=False, default='',
                        help='experiment name (default will be config folder name)')
    parser.add_argument('-i', '--index', type=int,
                        required=False, default=-1,
                        help='run a single experiment in the folder, specified by index')
    args = parser.parse_args()


    # Get JSON files
    config_dir = os.path.join(os.getcwd(), 'configs', args.config_dir)
    config_files = sorted([str(f) for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f))])
    assert len(config_files) > 0

    # Get experiment name
    exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_dir
    print('Config folder:\t {}'.format(exp_name))

    # Get save directory
    save_dir = os.path.join(os.getcwd(), args.save_dir, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'configs'))
    print('Save directory:\t {}'.format(save_dir))

    # Get device ID
    if torch.cuda.is_available():
        device = torch.cuda.device_count()
    else:
        device = 'cpu'
    print('Device:\t {}'.format(device))

    #init master
    master = manager.dict()
    master['start_time'] = strftime("%Y-%m-%dT%H-%M-%S", gmtime())
    master['experiment_name'] = exp_name
    master['device'] = device
    master['summaries'] = {}

    # Run a single experiment
    if args.index >= 0:
        if args.index < len(config_files):
            config_files = [config_files[args.index]]
        else:
            print("WARNING: Index out of range, will run all experiments in folder.")

    if device == 'cpu':
        inputs = [(config_file, config_dir, save_dir, exp_name, device, args, master) for config_file in config_files]
    else:
        inputs = []
        for dct in range(len(config_files)):
            inputs.append((config_files[dct], config_dir, save_dir, exp_name, dct%device, args, master))
    procs = []
    for input_args in inputs:
        p = Process(target=run_config, args=(input_args,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
