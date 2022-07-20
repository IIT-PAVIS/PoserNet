"""
Name: hyperparameters_management.py
Description: Contains a function that sets the hyperparameters either to default values or
             to the values contained in a configuration file.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import wandb
import argparse
import yaml
from core.pose_loss import PoseLoss
import torch
import numpy as np
import random
import git
import sys
import os


def set_experiment_parameters_and_log_them(c, f):
    """
    Sets the default values for the parameters and, in case, overwrites them with those specified in a configuration file (yaml).

    Some of the values are fixed (cannot be set via the configuration file), some are set according to information from the
    environment (current Git branch, if applicable).
    """
    # Get paths configuration from paths_config.yaml
    with open('paths_config.yaml', 'r') as stream:
        try:
            paths_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print('Terminating the program.', flush=True)
            sys.exit(-1)

    hyperparameter_defaults = dict(
        run_name='Default experiment',
        experiment_type='small_graphs_normal',
        dataset_path=paths_config['dataset_path'],
        model_output_path=paths_config['model_output_path'],
        n_epochs=10,
        perform_node_updates=True,
        shuffle_training_examples=True,
        corrupt_input_estimates_at_beginning_of_epoch=True,
        type_of_input_estimates=2,  # 0: Corrupted GT, 1: Estimates from matching BB centres, 2: Estimates from matching keypoints inside BBs, 3: Random poses, 4: Estimated from matching keypoints over the whole image.
        label_for_type_of_input=['Corrupted GT',
                                 'Estimates from matching BB centres',
                                 'Estimates from matching keypoints inside BBs',
                                 'Random poses',
                                 'Estimates from matching keypoints over the whole image'],
        load_data_from_binary_files=True,
        save_data_to_binary_files=False,

        input_file_name='eccv_final_objectness_set',
        left_out_scene_id=None,
        learning_rate=1e-3,
        use_lr_scheduler=True,
        lr_scheduler_factor=0.316,  # When applied twice, reduces the LR by a factor of 10.
        lr_patience=3,  # After 3 epochs with no improvement on the validation loss, it lowers the LR.

        seed=0,
        config_file_name='',  # Will be overwritten, in case a config file was specified.
        training_batch_size=1  # This has to be 1, the current code (especially msg_passing.py) is not written to handle minibatches.
    )

    # Set defaults.
    wandb.config.update(hyperparameter_defaults)

    # Optional: get hyperparameters from a config file and overwrite configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='', metavar='', help='Configuration file')
    config_file_name = (parser.parse_args()).config_file
    if config_file_name != '':
        with open(config_file_name, 'r') as stream:
            try:
                hyperparams_from_file = yaml.safe_load(stream)
                hyperparams_from_file['config_file_name'] = config_file_name
                # Update configuration with parameters set in config file.
                wandb.config.update(hyperparams_from_file, allow_val_change=True)
                print('Using configuration from: {:s}.'.format(config_file_name))

            except yaml.YAMLError as exc:
                print(exc)
                print('Terminating the program.', flush=True)
                sys.exit(-1)
    else:
        print('Config file not specified, using the default values.')

    wandb.run.name = c.run_name

    if c.experiment_type != 'small_graphs_normal' and c.experiment_type != 'small_graphs_loo' and c.experiment_type != 'large_graphs_normal':
        print('You specified an experiment type that is not recognised: {:s}.'.format(c.experiment_type))
        print('Terminating the program.', flush=True)
        sys.exit(-1)

    ############################################
    # Set loss function, optimiser, scheduler. #
    # These are fixed, not affected by config. #
    ############################################
    f['loss_function'] = PoseLoss()
    c.loss_function = f['loss_function']

    f['optimizer_function'] = torch.optim.Adam
    c.optimizer_function = f['optimizer_function']

    f['lr_scheduler_function'] = torch.optim.lr_scheduler.ReduceLROnPlateau
    c.lr_scheduler_function = f['lr_scheduler_function']

    ###############################################
    # Set variables based on current environment. #
    ###############################################
    # Log the current Git branch and the latest commit.
    try:
        c.git_branch = git.Repo(os.getcwd()).head.reference.name
        c.git_commit = git.Repo(os.getcwd()).head.reference.commit.message
    except:
        c.git_branch = 'N.A.'
        c.git_commit = 'N.A.'

    ####################################################
    # Set randomisation seed for reproducible results. #
    ####################################################
    torch.manual_seed(c.seed)  # Set seed for Torch.
    random.seed(c.seed)  # Set seed for Random. We probably do not use this.
    np.random.seed(c.seed)  # Set seed for NumPy.

    assert (c.training_batch_size == 1)  # This has to be 1, the current code (especially msg_passing.py) is not written to handle minibatches.
