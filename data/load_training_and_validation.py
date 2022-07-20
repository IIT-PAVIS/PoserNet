"""
Name: load_training_and_validation.py
Description: Top-level function for loading the data for different types of experiments.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import sys
from data.seven_scenes_dataset import get_training_data_for_all_scenes, get_training_data_for_all_scenes_leave_one_out, \
                                      get_validation_data_for_all_scenes, get_validation_data_for_all_scenes_leave_one_out


def load_training_and_validation(experiment_type, dataset_path, load_data_from_binary_files, label_for_type_of_input,
                                 type_of_input_estimates, input_file_name, save_data_to_binary_files, left_out_scene_id):
    if experiment_type == 'small_graphs_loo':
        print('Experiment type: small graphs, leave one scene out.', flush=True)
        print('Type of input data for edges: {:s}.'.format(label_for_type_of_input[type_of_input_estimates]), flush=True)
        print('Loading training data.', flush=True)
        training_data_list = get_training_data_for_all_scenes_leave_one_out(dataset_root_dir=dataset_path,
                                                                            sequence_name=input_file_name,
                                                                            load_data_from_binary_files=load_data_from_binary_files,
                                                                            save_data_to_binary_files=save_data_to_binary_files,
                                                                            type_of_input_estimates=type_of_input_estimates,
                                                                            left_out_scene_id=left_out_scene_id)
        print('Loading validation data.', flush=True)
        validation_data_list = get_validation_data_for_all_scenes_leave_one_out(dataset_root_dir=dataset_path,
                                                                                sequence_name=input_file_name,
                                                                                load_data_from_binary_files=load_data_from_binary_files,
                                                                                save_data_to_binary_files=save_data_to_binary_files,
                                                                                type_of_input_estimates=type_of_input_estimates,
                                                                                left_out_scene_id=left_out_scene_id)
    elif experiment_type == 'small_graphs_normal' or 'large_graphs_normal':
        if experiment_type == 'small_graphs_normal':
            print('Experiment type: small graphs, normal.', flush=True)
        else:
            print('Experiment type: large graphs, normal.', flush=True)
        print('Type of input data for edges: {:s}.'.format(label_for_type_of_input[type_of_input_estimates]), flush=True)
        print('Loading training data.', flush=True)
        training_data_list = get_training_data_for_all_scenes(dataset_root_dir=dataset_path,
                                                              sequence_name=input_file_name,
                                                              load_data_from_binary_files=load_data_from_binary_files,
                                                              save_data_to_binary_files=save_data_to_binary_files,
                                                              type_of_input_estimates=type_of_input_estimates)

        print('Loading validation data.', flush=True)
        validation_data_list = get_validation_data_for_all_scenes(dataset_root_dir=dataset_path,
                                                                  sequence_name=input_file_name,
                                                                  load_data_from_binary_files=load_data_from_binary_files,
                                                                  save_data_to_binary_files=save_data_to_binary_files,
                                                                  type_of_input_estimates=type_of_input_estimates)
    else:
        print('You requested a dataset that was not recognised {:s}.'.format(experiment_type), flush=True)
        print('Terminating the program.', flush=True)
        sys.exit(-1)

    return training_data_list, validation_data_list
