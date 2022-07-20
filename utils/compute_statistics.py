"""
Name: compute_statistics.py
Description: Functions used to compute statistics on the performance of PoserNet for one experiment.
             The functions also compute and store the corresponding plots.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored
from matplotlib.patches import Rectangle


def smooth_with_rolling_avg(data, window_length):
    mean = np.zeros(len(data) - window_length)
    stddev = np.zeros(len(data) - window_length)
    for index in range(len(data) - window_length):
        mean[index] = np.mean(data[index:index + window_length])
        stddev[index] = np.std(data[index:index + window_length])
    return mean, stddev



def compute_statistics(data, print_data=False):
    thresholds = [3, 5, 10, 30, 45]
    n_examples = float(data['orientation_errors'].shape[0])

    if print_data:
        print('ORIENTATION')
    percentages_below_thresholds_orientation = np.zeros(len(thresholds))
    for index, threshold in enumerate(thresholds):
        percentages_below_thresholds_orientation[index] = float(sum(np.rad2deg(data['orientation_errors']) < threshold) * 100.0 / n_examples)
        if print_data:
            print('orientation, percentage below {:d} = {:0.2f}'.format(threshold, percentages_below_thresholds_orientation[index]))
    percentages_below_thresholds_translation_direction = np.zeros(len(thresholds))
    # Population stats on orientation
    mean_orientation = np.rad2deg(np.mean(data['orientation_errors']))
    median_orientation = np.rad2deg(np.median(data['orientation_errors']))
    std_orientation = np.rad2deg(np.std(data['orientation_errors']))
    if print_data:
        print(' ')
        print('orientation median = {:0.2f}'.format(median_orientation))
        print(' ')


    if print_data:
        print(colored('TRANSLATION', 'blue'))
    for index, threshold in enumerate(thresholds):
        percentages_below_thresholds_translation_direction[index] = float(sum(np.rad2deg(data['translation_direction_errors']) < threshold) * 100.0 / n_examples)
        if print_data:
            print(colored('translation, percentage below {:d} = {:0.2f}'.format(threshold, percentages_below_thresholds_translation_direction[index]), 'blue'))
    # Population stats on translation direction
    mean_translation_direction = np.rad2deg(np.mean(data['translation_direction_errors']))
    median_translation_direction = np.rad2deg(np.median(data['translation_direction_errors']))
    std_translation_direction = np.rad2deg(np.std(data['translation_direction_errors']))
    if print_data:
        print(' ')
        print(colored('translation_direction median = {:0.2f}'.format(median_translation_direction), 'blue'))
        print(' ')

    return percentages_below_thresholds_orientation, percentages_below_thresholds_translation_direction, mean_orientation, median_orientation, std_orientation, mean_translation_direction, median_translation_direction, std_translation_direction


def compute_error_statistics_and_plot(experiment_label, raw_losses_file, losses_after_pn_files, output_directory, compute_smoothing=True):
    window_length = 50  # Length of the window used to compute rolling mean and rolling std dev.
    n_runs = len(losses_after_pn_files)

    ###########################################
    # Compute error statistics on input data. #
    ###########################################
    data = np.load(raw_losses_file)
    input_stats_orientation, input_stats_translation_direction, input_mean_orientation, input_median_orientation, input_std_orientation, input_mean_translation_direction, input_median_translation_direction, input_std_translation_direction = compute_statistics(data, print_data=False)
    print('Input data.')
    print('\tOrientation - Percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(input_stats_orientation[0], input_stats_orientation[1], input_stats_orientation[2], input_stats_orientation[3], input_stats_orientation[4]))
    print('\tOrientation - Median = {:.2f}'.format(input_median_orientation))
    print(' ')
    print(colored('\tTransl. dir. - Percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(input_stats_translation_direction[0], input_stats_translation_direction[1], input_stats_translation_direction[2], input_stats_translation_direction[3], input_stats_translation_direction[4]), 'blue'))
    print(colored('\tTransl. dir. - Median = {:.2f}'.format(input_median_translation_direction), 'blue'))



    initial_orient_errors = data['orientation_errors']
    initial_transl_errors = data['translation_direction_errors']
    my_bins = np.arange(0, 180, 1)

    plt.figure(1)
    plt.hist(np.rad2deg(initial_orient_errors), bins=my_bins, label='Initial')
    plt.figure(3)
    plt.hist(np.rad2deg(initial_transl_errors), bins=my_bins, label='Initial')

    stats_orientation = np.zeros((n_runs, 5))
    stats_translation_direction = np.zeros((n_runs, 5))
    mean_orientation = np.zeros(n_runs)
    median_orientation = np.zeros(n_runs)
    std_orientation = np.zeros(n_runs)
    mean_translation_direction = np.zeros(n_runs)
    median_translation_direction = np.zeros(n_runs)
    std_translation_direction = np.zeros(n_runs)

    posernet_orient_errors = 0
    posernet_transl_errors = 0
    for index, losses_after_pn_file in enumerate(losses_after_pn_files):
        #############################################################
        # Compute error statistics for one run, on the output data. #
        #############################################################
        data = np.load(losses_after_pn_file)
        stats_orientation[index, :], stats_translation_direction[index, :], mean_orientation[index], median_orientation[index], std_orientation[index], mean_translation_direction[index], median_translation_direction[index], std_translation_direction[index] = compute_statistics(data)
        posernet_orient_errors += data['orientation_errors']
        posernet_transl_errors += data['translation_direction_errors']

    posernet_orient_errors /= float(n_runs)
    posernet_transl_errors /= float(n_runs)

    posernet_orient_errors = np.rad2deg(posernet_orient_errors.squeeze())
    initial_orient_errors = np.rad2deg(initial_orient_errors.squeeze())
    posernet_transl_errors = np.rad2deg(posernet_transl_errors.squeeze())
    initial_transl_errors = np.rad2deg(initial_transl_errors.squeeze())


    plt.figure(1)
    plt.hist(posernet_orient_errors, bins=my_bins, alpha=0.5, label='PoserNet')
    plt.title('{:s} - Orientation error'.format(experiment_label))
    plt.xlabel('Orientation error [deg]')
    plt.ylabel('Frequency')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [(0.12, 0.47, 0.70), (1.00, 0.50, 0.05)]]
    labels = ['Initial', 'PoserNet']
    plt.legend(handles, labels)
    output_fig_file_name = '{:s}/{:s}_histogram_of_orientation_error.png'.format(output_directory, experiment_label)
    plt.savefig(output_fig_file_name, dpi=300)

    plt.figure(3)
    plt.hist(posernet_transl_errors, bins=my_bins, alpha=0.5, label='PoserNet')
    plt.title('{:s} - Translation direction error'.format(experiment_label))
    plt.xlabel('Translation direction error [deg]')
    plt.ylabel('Frequency')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [(0.12, 0.47, 0.70), (1.00, 0.50, 0.05)]]
    labels = ['Initial', 'PoserNet']
    plt.legend(handles, labels)
    output_fig_file_name = '{:s}/{:s}_histogram_of_translation_direction_error.png'.format(output_directory, experiment_label)
    plt.savefig(output_fig_file_name, dpi=300)

    means_orientation = np.mean(stats_orientation, 0)
    stds_orientation = np.std(stats_orientation, 0)
    means_translation_direction = np.mean(stats_translation_direction, 0)
    stds_translation_direction = np.std(stats_translation_direction, 0)

    print(' ')
    print('Output data:')
    print('\tOrientation - Mean percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(means_orientation[0], means_orientation[1], means_orientation[2], means_orientation[3], means_orientation[4]))
    print('\tOrientation - Std of percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(stds_orientation[0], stds_orientation[1], stds_orientation[2], stds_orientation[3], stds_orientation[4]))
    print(' ')
    print('\tOrientation - Mean of medians = {:.2f}'.format(np.mean(median_orientation)))
    print('\tOrientation - Std of medians = {:.2f}'.format(np.std(median_orientation)))
    print(' ')
    print(colored('\tTransl. dir. - Mean percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(means_translation_direction[0], means_translation_direction[1], means_translation_direction[2], means_translation_direction[3], means_translation_direction[4]), 'blue'))
    print(colored('\tTransl. dir. - Std of percentage below thresholds = {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(stds_translation_direction[0], stds_translation_direction[1], stds_translation_direction[2], stds_translation_direction[3], stds_translation_direction[4]), 'blue'))
    print(' ')
    print(colored('\tTransl. dir. - Mean of medians = {:.2f}'.format(np.mean(median_translation_direction)), 'blue'))
    print(colored('\tTransl. dir. - Std of medians = {:.2f}'.format(np.std(median_translation_direction)), 'blue'))

    # Line plots, Orientation error
    plt.figure()
    indices = posernet_orient_errors.argsort()
    posernet_orient_errors = posernet_orient_errors[indices]
    initial_orient_errors = initial_orient_errors[indices]
    if compute_smoothing:
        initial_orient_errors_mean, initial_orient_errors_stddev = smooth_with_rolling_avg(initial_orient_errors, window_length)
    else:
        initial_orient_errors_mean = initial_orient_errors

    plt.plot(initial_orient_errors_mean)
    if compute_smoothing:
        plt.fill_between(range(len(initial_orient_errors_mean)), initial_orient_errors_mean - initial_orient_errors_stddev, initial_orient_errors_mean + initial_orient_errors_stddev, alpha=.3)
    plt.plot(posernet_orient_errors)
    plt.title('{:s} - Orientation error'.format(experiment_label))
    plt.xlabel('Graph index (sorted)')
    plt.ylabel('Orientation error [deg]')
    plt.legend(['Initial', 'PoserNet'])

    output_fig_file_name = '{:s}/{:s}_sorted_orientation_errors.png'.format(output_directory, experiment_label)
    plt.savefig(output_fig_file_name)

    # Line plots, Translation direction error
    plt.figure()
    indices = posernet_transl_errors.argsort()
    posernet_transl_errors = posernet_transl_errors[indices]
    initial_transl_errors = initial_transl_errors[indices]
    if compute_smoothing:
        initial_transl_errors_mean, initial_transl_errors_stddev = smooth_with_rolling_avg(initial_transl_errors, window_length)
    else:
        initial_transl_errors_mean = initial_transl_errors

    plt.plot(initial_transl_errors_mean)
    if compute_smoothing:
        plt.fill_between(range(len(initial_transl_errors_mean)), initial_transl_errors_mean - initial_transl_errors_stddev, initial_transl_errors_mean + initial_transl_errors_stddev, alpha=.3)
    plt.plot(posernet_transl_errors)
    plt.title('{:s} - Translation direction error'.format(experiment_label))
    plt.xlabel('Graph index (sorted)')
    plt.ylabel('Translation direction error [deg]')
    plt.legend(['Initial', 'PoserNet'])

    output_fig_file_name = '{:s}/{:s}_sorted_translation_direction_errors.png'.format(output_directory, experiment_label)
    plt.savefig(output_fig_file_name)

    plt.close('all')