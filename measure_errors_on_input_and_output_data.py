"""
Name: measure_errors_on_input_and_output_data.py
Description: Script used to assess the performance of PoserNet models on input and output data.
             The results are stored and will be loaded automatically by compute_error_statistics_for_tables.py.
             Before running this script, it is required that experiments have been run (run_experiments.sh).
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
from utils.compute_and_save_losses import compute_and_save_losses


# Small graphs, normal.
models_to_be_evaluated = [{'model_id': 'Small_graphs_normal_000000_seed_1', 'experiment_configuration_file': 'small_graphs_normal_000000.yaml'},
                          {'model_id': 'Small_graphs_normal_000001_seed_2', 'experiment_configuration_file': 'small_graphs_normal_000001.yaml'},
                          {'model_id': 'Small_graphs_normal_000002_seed_3', 'experiment_configuration_file': 'small_graphs_normal_000002.yaml'}]
compute_and_save_losses(models_to_be_evaluated, load_data_from_binary_files=True)

# Small graphs, loo.
models_to_be_evaluated = [{'model_id': 'Small_graphs_loo_000000_left_out_scene_3', 'experiment_configuration_file': 'small_graphs_loo_000000.yaml'},
                          {'model_id': 'Small_graphs_loo_000001_left_out_scene_4', 'experiment_configuration_file': 'small_graphs_loo_000001.yaml'},
                          {'model_id': 'Small_graphs_loo_000002_left_out_scene_5', 'experiment_configuration_file': 'small_graphs_loo_000002.yaml'},
                          {'model_id': 'Small_graphs_loo_000003_left_out_scene_6', 'experiment_configuration_file': 'small_graphs_loo_000003.yaml'},
                          {'model_id': 'Small_graphs_loo_000004_left_out_scene_0', 'experiment_configuration_file': 'small_graphs_loo_000004.yaml'},
                          {'model_id': 'Small_graphs_loo_000005_left_out_scene_1', 'experiment_configuration_file': 'small_graphs_loo_000005.yaml'},
                          {'model_id': 'Small_graphs_loo_000006_left_out_scene_2', 'experiment_configuration_file': 'small_graphs_loo_000006.yaml'}]
compute_and_save_losses(models_to_be_evaluated, load_data_from_binary_files=True)

# Large graphs, normal.
models_to_be_evaluated = [{'model_id': 'Large_graphs_normal_000000_seed_1', 'experiment_configuration_file': 'large_graphs_normal_000000.yaml'},
                          {'model_id': 'Large_graphs_normal_000001_seed_2', 'experiment_configuration_file': 'large_graphs_normal_000001.yaml'},
                          {'model_id': 'Large_graphs_normal_000002_seed_3', 'experiment_configuration_file': 'large_graphs_normal_000002.yaml'}]
compute_and_save_losses(models_to_be_evaluated, load_data_from_binary_files=True)
