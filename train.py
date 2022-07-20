"""
Name: train.py
Description: Training script for PoserNet.
             Example of usage: python train.py --config-file experiment_configs/large_graphs_normal_000001.yaml.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""

# Generic imports.
from pathlib import Path

# Deep Learning imports.
import torch

# Logger
import wandb
wandb.init(project="PoserNet", entity="matteo_taiana")  # Set this to your own project and entity.

# Project imports.
from core.pose_refiner import PoseRefiner0
from utils.hyperparameter_management import set_experiment_parameters_and_log_them

# Data loading imports and functions.
from data.load_training_and_validation import load_training_and_validation
from data.seven_scenes_dataset import corrupt_training_set
from data.my_dataloader import DataLoader

# The data structures used in the graphs are described my_graph.py.


def main():
    # Create an alias for the configuration object.
    c = wandb.config
    # This is the dictionary which contains several functions: loss, optimizer, scheduler.
    f = {}
    # Set hyperparameters: a combination of default values, values specified in a config file and values computed based on the environment.
    set_experiment_parameters_and_log_them(c, f)

    ##########################
    # Prepare training data. #
    ##########################
    # Load data.
    training_data_list, validation_data_list = load_training_and_validation(c.experiment_type, c.dataset_path, c.load_data_from_binary_files,
                                                                            c.label_for_type_of_input, c.type_of_input_estimates,
                                                                            c.input_file_name, c.save_data_to_binary_files, c.left_out_scene_id)

    # Make sure the data is on the GPU
    print('Pushing data to GPU.', flush=True)
    for graph in training_data_list:
        graph.cuda()
    for graph in validation_data_list:
        graph.cuda()

    print('Setting up data loaders.', flush=True)
    training_loader = DataLoader(training_data_list, batch_size=c.training_batch_size, shuffle=c.shuffle_training_examples)
    validation_loader = DataLoader(validation_data_list, batch_size=1)
    print('{:d} graphs will be used for training.'.format(len(training_data_list)), flush=True)
    print('{:d} graphs will be used for validation/testing.'.format(len(validation_data_list)), flush=True)



    ########
    # Main #
    ########
    model = PoseRefiner0(c.perform_node_updates, c.type_of_input_estimates)
    model = model.cuda()
    wandb.watch(model, log='all', log_freq=10)

    optimizer = f['optimizer_function'](model.parameters(), lr=c.learning_rate)
    if c.use_lr_scheduler:
        # TODO: make this automatic.
        scheduler = f['lr_scheduler_function'](optimizer, factor=c.lr_scheduler_factor, patience=c.lr_patience)
        # scheduler = f['lr_scheduler_function'](optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Make sure directory for storing output models exists.
    Path(c.model_output_path).mkdir(parents=True, exist_ok=True)
    output_model_dir = '{:s}/Run_{:s}'.format(c.model_output_path, wandb.run.name)
    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    print('Model will be saved to: {:s}.'.format(output_model_dir), flush=True)

    print(' ', flush=True)
    best_validation_loss = 99999999
    training_reporting_period = 1000  # Log and display moving average at intervals of this length.
    for epoch in range(c.n_epochs):
        # Augmentation: corrupt the input data with a little noise 4 times out of 5
        if c.corrupt_input_estimates_at_beginning_of_epoch and not (epoch % 5 == 0):
            current_training_data_list = corrupt_training_set(training_data_list, translation_noise_limit=0.033,
                                                              rotation_noise_limit_deg=2)
        else:
            current_training_data_list = training_data_list
        training_loader = DataLoader(current_training_data_list, batch_size=c.training_batch_size,
                                     shuffle=c.shuffle_training_examples)

        print('***** Training ***********************************************************************************************', flush=True)
        model.train()
        epoch_training_loss = 0
        epoch_training_quat_dir_loss = 0
        epoch_training_transl_dir_loss = 0
        epoch_training_quat_norm_loss = 0
        epoch_training_transl_norm_loss = 0
        batch_index = 0
        for data in training_loader:
            optimizer.zero_grad()
            output = model(data)
            [total_loss,
             quat_dir_loss,
             transl_dir_loss,
             quat_norm_loss,
             transl_norm_loss] = f['loss_function'](output, data.y)

            total_loss.backward()

            # Make sure the gradients do not explode
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)

            optimizer.step()

            optimizer.zero_grad()
            epoch_training_loss += float(total_loss.item())
            epoch_training_quat_dir_loss += float(quat_dir_loss.item())
            epoch_training_transl_dir_loss += float(transl_dir_loss.item())
            epoch_training_quat_norm_loss += float(quat_norm_loss.item())
            epoch_training_transl_norm_loss += float(transl_norm_loss.item())

            if epoch == 0 and batch_index < 10:
                print('Batch [{:5d}]. Instant training loss =    {:.3f}, '
                      'q_dir = {:.3f}, '
                      't_dir = {:.3f}, '
                      'q_norm = {:.3f}, '
                      't_norm = {:.3f}. '.format(batch_index,
                                                 total_loss.cpu().detach().item(),
                                                 quat_dir_loss.cpu().detach().item(),
                                                 transl_dir_loss.cpu().detach().item(),
                                                 quat_norm_loss.cpu().detach().item(),
                                                 transl_norm_loss.cpu().detach().item()), flush=True)
            if epoch == 0 and batch_index == 9:
                print('--------------------------------------------------------------------------------------------------------------', flush=True)

            n_processed_batches = batch_index + 1
            if batch_index > 0 and n_processed_batches % training_reporting_period == 0:
                print('Batch [{:5d}]. Local avg. training loss = {:.3f}, '
                      'q_dir = {:.3f}, '
                      't_dir = {:.3f}, '
                      'q_norm = {:.3f}, '
                      't_norm = {:.3f}. '.format(batch_index,
                                                 cum_total_loss/training_reporting_period,
                                                 cum_quat_dir_loss/training_reporting_period,
                                                 cum_transl_dir_loss/training_reporting_period,
                                                 cum_quat_norm_loss/training_reporting_period,
                                                 cum_transl_norm_loss/training_reporting_period), flush=True)
                wandb.log({'mov_avg_training_loss': cum_total_loss / training_reporting_period, 'epoch': epoch})
                wandb.log({'mov_avg_training_quat_dir_loss': cum_quat_dir_loss / training_reporting_period, 'epoch': epoch})
                wandb.log({'mov_avg_training_transl_dir_loss': cum_transl_dir_loss / training_reporting_period, 'epoch': epoch})
                wandb.log({'mov_avg_training_quat_norm_loss': cum_quat_norm_loss / training_reporting_period, 'epoch': epoch})
                wandb.log({'mov_avg_training_transl_norm_loss': cum_transl_norm_loss / training_reporting_period, 'epoch': epoch})

            if batch_index % training_reporting_period == 0:
                cum_total_loss = 0
                cum_quat_dir_loss = 0
                cum_transl_dir_loss = 0
                cum_quat_norm_loss = 0
                cum_transl_norm_loss = 0
            else:
                cum_total_loss       += float(total_loss.item())
                cum_quat_dir_loss    += float(quat_dir_loss.item())
                cum_transl_dir_loss  += float(transl_dir_loss.item())
                cum_quat_norm_loss   += float(quat_norm_loss.item())
                cum_transl_norm_loss += float(transl_norm_loss.item())

            batch_index += 1

        wandb.log({'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch})

        print('***** Validation *********************************************************************************************', flush=True)
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = 0
            epoch_validation_quat_dir_loss = 0
            epoch_validation_transl_dir_loss = 0
            epoch_validation_quat_norm_loss = 0
            epoch_validation_transl_norm_loss = 0
            n_validation_batches = 0
            for data_val in validation_loader:
                output_val = model(data_val)
                [total_loss,
                 quat_dir_loss,
                 transl_dir_loss,
                 quat_norm_loss,
                 transl_norm_loss] = f['loss_function'](output_val, data_val.y)
                epoch_validation_loss += float(total_loss.item())
                epoch_validation_quat_dir_loss += float(quat_dir_loss.item())
                epoch_validation_transl_dir_loss += float(transl_dir_loss.item())
                epoch_validation_quat_norm_loss += float(quat_norm_loss.item())
                epoch_validation_transl_norm_loss += float(transl_norm_loss.item())

                n_validation_batches += 1

            # Compute average losses for the training and the validation set, print them and store them in Sacred.
            epoch_training_loss /= batch_index
            epoch_training_quat_dir_loss /= batch_index
            epoch_training_transl_dir_loss /= batch_index
            epoch_training_quat_norm_loss /= batch_index
            epoch_training_transl_norm_loss /= batch_index

            epoch_validation_loss /= n_validation_batches
            epoch_validation_quat_dir_loss /= n_validation_batches
            epoch_validation_transl_dir_loss /= n_validation_batches
            epoch_validation_quat_norm_loss /= n_validation_batches
            epoch_validation_transl_norm_loss /= n_validation_batches


            print('Epoch [{:3d}]. Training loss   = {:.3f}, '
                  'q_dir = {:.3f}, '
                  't_dir = {:.3f}, '
                  'q_norm = {:.3f}, '
                  't_norm = {:.3f}. '.format(epoch,
                                             epoch_training_loss,
                                             epoch_training_quat_dir_loss,
                                             epoch_training_transl_dir_loss,
                                             epoch_training_quat_norm_loss,
                                             epoch_training_transl_norm_loss), flush=True)
            print('Epoch [{:3d}]. Validation loss = {:.3f}, '
                  'q_dir = {:.3f}, '
                  't_dir = {:.3f}, '
                  'q_norm = {:.3f}, '
                  't_norm = {:.3f}. '.format(epoch,
                                             epoch_validation_loss,
                                             epoch_validation_quat_dir_loss,
                                             epoch_validation_transl_dir_loss,
                                             epoch_validation_quat_norm_loss,
                                             epoch_validation_transl_norm_loss))
            print('**************************************************************************************************************', flush=True)
            print(' ', flush=True)

            wandb.log({'epoch_training_loss': epoch_training_loss, 'epoch': epoch})
            wandb.log({'epoch_training_quat_dir_loss': epoch_training_quat_dir_loss, 'epoch': epoch})
            wandb.log({'epoch_training_transl_dir_loss': epoch_training_transl_dir_loss, 'epoch': epoch})
            wandb.log({'epoch_training_quat_norm_loss': epoch_training_quat_norm_loss, 'epoch': epoch})
            wandb.log({'epoch_training_transl_norm_loss': epoch_training_transl_norm_loss, 'epoch': epoch})

            wandb.log({'epoch_validation_loss': epoch_validation_loss, 'epoch': epoch})
            wandb.log({'epoch_validation_quat_dir_loss': epoch_validation_quat_dir_loss, 'epoch': epoch})
            wandb.log({'epoch_validation_transl_dir_loss': epoch_validation_transl_dir_loss, 'epoch': epoch})
            wandb.log({'epoch_validation_quat_norm_loss': epoch_validation_quat_norm_loss, 'epoch': epoch})
            wandb.log({'epoch_validation_transl_norm_loss': epoch_validation_transl_norm_loss, 'epoch': epoch})

            if c.use_lr_scheduler:
                # TODO: make this change automatically based on the selected scheduler
                # Reduce the learning rate when validation loss does not decrease for N consecutive epochs.
                scheduler.step(epoch_validation_loss)
                # scheduler.step(epoch_validation_loss)  # For all the other schedulers.

            # When the current model has the best performance on the validation data so far, store it.
            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                training_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(training_state, '{:s}/training_state.pth'.format(output_model_dir))
                with open('{:s}/info.txt'.format(output_model_dir), 'w') as out_file:
                    out_file.write('Epoch ={:d}.\nEpoch training loss = {:f}.\nEpoch validation loss = {:f}.\n'.format(
                        epoch,
                        epoch_training_loss,
                        epoch_validation_loss))



if __name__ == "__main__":
    main()
