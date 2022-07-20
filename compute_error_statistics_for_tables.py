"""
Name: compute_error_statistics_for_tables.py
Description: Script used to compute performance metrics reported in the paper for three different kinds of experiments.
             It also produces and stores plots that are useful to analysing the performance of PoserNet.
             Before running this script, it is required that experiments have been run (run_experiments.sh) and that the
             performance of each experiment has been computed and stored (measure_errors_on_input_and_output_data.py).
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import yaml
import sys
from pathlib import Path
from utils.compute_statistics import compute_error_statistics_and_plot


# Get paths configuration from paths_config.yaml
with open('paths_config.yaml', 'r') as stream:
    try:
        paths_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        print('Terminating the program.', flush=True)
        sys.exit(-1)
stats_directory = paths_config['dataset_path'] + '/output_for_stats/'
plots_directory = paths_config['dataset_path'] + '/plots/'
Path(plots_directory).mkdir(parents=True, exist_ok=True)


############################
# 1. Small graphs, normal. #
############################
print('**************************************************************************************************************', flush=True)
print('Performance evaluation for the small_graphs_normal experiment.')
experiment_label = 'small_graphs_normal'
raw_losses_file = stats_directory + '/eccv_final_objectness_set_keypoints_input.npz'
losses_after_pn_files = [stats_directory + '/eccv_final_objectness_set_keypoints_output_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_seed_2.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_seed_3.npz']
compute_error_statistics_and_plot(experiment_label, raw_losses_file, losses_after_pn_files,
                                  plots_directory, compute_smoothing=True)
print(' ', flush=True)

############################
# 2. Large graphs, normal. #
############################
print('**************************************************************************************************************', flush=True)
print('Performance evaluation for the large_graphs_normal experiment.')
experiment_label = 'large_graphs_normal'
raw_losses_file = stats_directory + '/eccv_final_objectness_set_long_keypoints_input.npz'
losses_after_pn_files = [stats_directory + '/eccv_final_objectness_set_long_keypoints_output_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_long_keypoints_output_seed_2.npz',
                         stats_directory + '/eccv_final_objectness_set_long_keypoints_output_seed_3.npz']
compute_error_statistics_and_plot(experiment_label, raw_losses_file, losses_after_pn_files,
                                  plots_directory, compute_smoothing=False)  # Smoothing cannot be used for large graphs because the data series is not long enough.
print(' ', flush=True)

#########################
# 3. Small graphs, loo. #
#########################
print('**************************************************************************************************************', flush=True)
print('Performance evaluation for the small_graphs_loo experiment.')
experiment_label = 'small_graphs_loo'
raw_losses_file = stats_directory + '/eccv_final_objectness_set_keypoints_input_loo_3.npz'  # Warning: the values reported for the input are only relative to this scene.
losses_after_pn_files = [stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_3_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_4_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_5_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_6_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_0_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_1_seed_1.npz',
                         stats_directory + '/eccv_final_objectness_set_keypoints_output_loo_2_seed_1.npz']
compute_error_statistics_and_plot(experiment_label, raw_losses_file, losses_after_pn_files,
                                  plots_directory, compute_smoothing=True)
