"""
Name: compute_and_save_losses.py
Description: Functions used to apply PoserNet on input data and for storing the corresponding output errors.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
# Path hack used to import scripts from sibling directories.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Generic imports
from pathlib import Path
import yaml
import sys

# Deep Learning imports.
import torch
import numpy as np

# Project imports.
from core.pose_refiner import PoseRefiner0
from core.pose_loss import PoseLoss

# Data loading imports.
from data.seven_scenes_dataset import get_validation_data_for_all_scenes, get_validation_data_for_all_scenes_leave_one_out
from data.my_dataloader import DataLoader


def compute_and_save_losses(models_to_be_evaluated, load_data_from_binary_files):
    #################
    # Set things up #
    #################
    # Get paths configuration.
    with open('paths_config.yaml', 'r') as stream:
        try:
            paths_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print('Terminating the program.', flush=True)
            sys.exit(-1)

    model_path = paths_config['model_output_path']
    output_path_for_losses = paths_config['dataset_path'] + '/output_for_stats/'
    Path(output_path_for_losses).mkdir(parents=True, exist_ok=True)
    Path(output_path_for_losses).mkdir(parents=True, exist_ok=True)

    loss_function = PoseLoss()
    save_data_to_binary_files = False

    for current_model in models_to_be_evaluated:
        print('**************************************************************************************************************', flush=True)
        print('Working on experiment {:s}, model {:s}.'.format(current_model['experiment_configuration_file'], current_model['model_id']))
        experiment_configuration_file = 'experiment_configs/' + current_model['experiment_configuration_file']
        # Get experiment configuration.
        with open(experiment_configuration_file, 'r') as stream:
            try:
                exp_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                print('Terminating the program.', flush=True)
                sys.exit(-1)
    
        if exp_config['type_of_input_estimates'] == 0:
            type_of_input_label = 'corrupted'
        elif exp_config['type_of_input_estimates'] == 1:
            type_of_input_label = 'bb'
        elif exp_config['type_of_input_estimates'] == 2:
            type_of_input_label = 'keypoints'
        elif exp_config['type_of_input_estimates'] == 3:
            type_of_input_label = 'random'
        elif exp_config['type_of_input_estimates'] == 4:
            type_of_input_label = 'keypoints_over_whole_image'
    

    
        ########################
        # Load validation data #
        ########################
        print('Loading validation data.')
        if exp_config['experiment_type'] == 'small_graphs_loo':
            validation_data_list = get_validation_data_for_all_scenes_leave_one_out(dataset_root_dir=paths_config['dataset_path'],
                                                                                    sequence_name=exp_config['input_file_name'],
                                                                                    load_data_from_binary_files=load_data_from_binary_files,
                                                                                    save_data_to_binary_files=save_data_to_binary_files,
                                                                                    type_of_input_estimates=exp_config['type_of_input_estimates'],
                                                                                    left_out_scene_id=exp_config['left_out_scene_id'])
        else:
            validation_data_list = get_validation_data_for_all_scenes(dataset_root_dir=paths_config['dataset_path'],
                                                                      sequence_name=exp_config['input_file_name'],
                                                                      load_data_from_binary_files=load_data_from_binary_files,
                                                                      save_data_to_binary_files=save_data_to_binary_files,
                                                                      type_of_input_estimates=exp_config['type_of_input_estimates'])
    
        #################
        # Prepare data. #
        #################
        # Make sure the data is on the GPU
        print('Pushing data to GPU.')
        for graph in validation_data_list:
            graph.cuda()
    
        print('Setting up the data loaders.')
        validation_loader = DataLoader(validation_data_list, batch_size=1)
        print('{:d} graphs will be used for validation/testing.'.format(len(validation_data_list)))


        ########################
        # Load and apply model #
        ########################
        model = PoseRefiner0(exp_config['perform_node_updates'], exp_config['type_of_input_estimates'])
        checkpoint_path_dict_path = model_path + '/Run_{:s}/training_state.pth'.format(current_model['model_id'])

        print('Loading network weights from: {:s}'.format(checkpoint_path_dict_path))
        checkpoint = torch.load(checkpoint_path_dict_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        model.eval()


        ############################
        # Compute and store losses #
        ############################
        # Compute losses for input data. The model is not used. The output depends on the type of data that is loaded: BB's estimates, keypoints estimates, etc.
        work_on_input_data = True
        print('***** Computing losses on input data. ************************************************************************', flush=True)
        output_file_name_input_data = output_path_for_losses + exp_config['input_file_name'] + '_' + type_of_input_label + '_input'
        if exp_config['experiment_type'] == 'small_graphs_loo':
            output_file_name_input_data += '_loo_{:d}'.format(exp_config['left_out_scene_id'])
        apply_one_model(model, validation_loader, loss_function, work_on_input_data, output_file_name_input_data)

        # Compute losses after applying PoserNet.
        work_on_input_data = False
        print('***** Computing losses on ouput data. ************************************************************************', flush=True)
        output_file_name_estimates = output_path_for_losses + exp_config['input_file_name'] + '_' + type_of_input_label + '_output'
        if exp_config['experiment_type'] == 'small_graphs_loo':
            output_file_name_estimates += '_loo_{:d}_seed_{:d}'.format(exp_config['left_out_scene_id'], exp_config['seed'])
        else:
            output_file_name_estimates += '_seed_{:d}'.format(exp_config['seed'])
        apply_one_model(model, validation_loader, loss_function, work_on_input_data, output_file_name_estimates)
        print('**************************************************************************************************************', flush=True)
        print(' ')




def apply_one_model(model, validation_loader, loss_function, work_on_input_data, output_file_name):
    with torch.no_grad():
        # Collecting data for plotting and statistics
        n_examples = len(list(validation_loader))
        orientation_errors = np.zeros((n_examples, 1))
        translation_direction_errors = np.zeros((n_examples, 1))
        quat_norm_errors = np.zeros((n_examples, 1))
        transl_norm_errors = np.zeros((n_examples, 1))

        epoch_validation_loss = 0
        epoch_validation_quat_dir_loss = 0
        epoch_validation_transl_dir_loss = 0
        epoch_validation_quat_norm_loss = 0
        epoch_validation_transl_norm_loss = 0
        batch_index = 0
        for data_val in validation_loader:
            if batch_index % 1000 == 0:
                print('Working on batch {:5d}.'.format(batch_index))

            if work_on_input_data:
                [total_loss,
                 quat_dir_loss,
                 transl_dir_loss,
                 quat_norm_loss,
                 transl_norm_loss,
                 ] = loss_function(data_val.edge_attr, data_val.y)  # Use input values rather than output ones
            else:
                output = model(data_val)
                [total_loss,
                 quat_dir_loss,
                 transl_dir_loss,
                 quat_norm_loss,
                 transl_norm_loss,
                 ] = loss_function(output, data_val.y)

            orientation_errors[batch_index] = float(quat_dir_loss.item())
            translation_direction_errors[batch_index] = float(transl_dir_loss.item())
            quat_norm_errors[batch_index] = float(quat_norm_loss.item())
            transl_norm_errors[batch_index] = float(transl_norm_loss.item())

            epoch_validation_loss += float(total_loss.item())
            epoch_validation_quat_dir_loss += float(quat_dir_loss.item())
            epoch_validation_transl_dir_loss += float(transl_dir_loss.item())
            epoch_validation_quat_norm_loss += float(quat_norm_loss.item())
            epoch_validation_transl_norm_loss += float(transl_norm_loss.item())

            batch_index += 1

        n_batches = batch_index
        # Compute average losses for the training and the validation set, print them and store them in Sacred.
        epoch_validation_loss /= n_batches
        epoch_validation_quat_dir_loss /= n_batches
        epoch_validation_transl_dir_loss /= n_batches
        epoch_validation_quat_norm_loss /= n_batches
        epoch_validation_transl_norm_loss /= n_batches

        # Print losses
        print('Full validation loss    = {:0.3f}'.format(epoch_validation_loss))
        print('Quat. dir. loss         = {:0.3f}'.format(epoch_validation_quat_dir_loss))
        print('Quat. dir. loss [deg]   = {:0.1f}'.format(np.rad2deg(epoch_validation_quat_dir_loss)))
        print('Transl. dir. loss       = {:0.3f}'.format(epoch_validation_transl_dir_loss))
        print('Transl. dir. loss [deg] = {:0.1f}'.format(np.rad2deg(epoch_validation_transl_dir_loss)))
        print('Quat. norm. loss        = {:0.3f}'.format(epoch_validation_quat_norm_loss))
        print('Transl. norm. loss      = {:0.3f}'.format(epoch_validation_transl_norm_loss))

        # Save file with arrays of losses.
        output_file_name = output_file_name
        np.savez(output_file_name, orientation_errors=orientation_errors, translation_direction_errors=translation_direction_errors, quat_norm_errors=quat_norm_errors, transl_norm_errors=transl_norm_errors)
        print('Results saved to: {:s}.'.format(output_file_name))
