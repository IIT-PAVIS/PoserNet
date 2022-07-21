<p align="center">
  <img height="250" src="https://raw.githubusercontent.com/IIT-PAVIS/PoserNet/master/web/PoserNet_logo_and_name.svg?sanitize=true" />
</p>

## Introduction
The code in this repository is part of the paper:
<br>
**[PoserNet: Refining Relative Camera Poses Exploiting Object Detections (arXiv)](https://arxiv.org/abs/2207.09445)**
<br>
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/matteo-taiana'>Matteo Taiana</a>,
 <a  href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/matteo-toso'>Matteo Toso</a>,
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/stuart-james'>Stuart James</a> and
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/alessio-delbue'>Alessio Del Bue</a>.
<br>
Accepted for publication at  [ECCV 2022](https://eccv2022.ecva.net/).

**Abstract**<br> 
The estimation of the camera poses associated with a set of images commonly relies on feature matches between the images. In contrast, we are the first to address this challenge by using objectness regions to guide the pose estimation problem rather than explicit semantic object detections. We propose Pose Refiner Network (PoserNet) a light-weight Graph Neural Network to refine the approximate pair-wise relative camera poses. PoserNet exploits associations between the objectness regions - concisely expressed as bounding boxes - across multiple views to globally refine sparsely connected view graphs. We evaluate on the 7-Scenes dataset across varied sizes of graphs and show how this process can be beneficial to optimisation-based Motion Averaging algorithms improving the median error on the rotation by 62 degrees with respect to the initial estimates obtained based on bounding boxes. 

PoserNet is published under the MIT license.<br> 
If you use this code in your research, please acknowledge it as:

    @inproceedings{posernet_eccv2022,
    Title = {PoserNet: Refining Relative Camera Poses Exploiting Object Detections},
    Author = {Matteo Taiana and Matteo Toso and Stuart James and Alessio Del Bue},
    booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
    Year = {2022},
    }

## Project set up

We developed PoserNet on computers running Ubuntu, but we expect that it can be run on other operating systems.

### Clone the repository and create the Conda environment
    git clone git@github.com:IIT-PAVIS/PoserNet
    cd PoserNet
    conda env create -f PoserNet.yaml
    conda activate PoserNet
 
In our setup, we use cudatoolkit=10.2. 
Please, remember to activate the PoserNet Conda environment before running other commands listed in this guide.

### Set up Weights & Biases (for logging and plotting)
This project logs the data that is generated during training using Weights and Biases (http://wandb.ai/).
For PoserNet's code to run, you need to have a WandB account, have it set up on your computer (https://docs.wandb.ai/quickstart), and you need to set the WandB project and entity fields in train.py with your information:

    wandb.init(project="PoserNet", entity="matteo_taiana")

### Download the training and testing data
Create a directory to store the data in, e.g.:
    
    mkdir 7Scenes/

The graphs used to train and evaluate PoserNet can be downloaded from [here](https://drive.google.com/drive/folders/1y7M3fVt0XCaJOfSavb8-AA1g6QyoXzur?usp=sharing).
The file PoserNet_graphs.zip contains two JSON files, with the large and small graphs respectively. Unzip the file in the chosen dataset directory.

### Set up input and output directories
Please set the path from which PoserNet will read the input data and that where it will store the checkpoints inside paths_config.yaml:

    dataset_path: '7Scenes/'
    model_output_path: 'PoserNetModels/'

PoserNet will automatically create these subdirectories for storing a binary version of the input data (which is much faster to load), information on the evaluation of PoserNet that is used to compute performance metrics and some plots:

    $dataset_path/cached_binary_input_data/
    $dataset_path/output_for_stats/
    $dataset_path/plots/

### Computing, storing and using binary input files  
Experiment configurations are stored in files like experiment_configs/small_graphs_normal_000000.yaml. 
The first time you run an experiment on some data, you will have to read the data from the JSON input file and save the binary version of that file (which is much faster to load, in the following runs). 
This is achieved by setting **load_data_from_binary_files: False**, and **save_data_to_binary_files: True** in the experiment configuration file.
The following times, you should run the train script with **load_data_from_binary_files: True** and **save_data_to_binary_files: False**.

## Running PoserNet
### Main scripts and project directory structure
The main scripts are located in the root directory:
* **train.py** is used for training PoserNet.
* **measure_errors_on_input_and_output_data.py** is used for measuring the performance of PoserNet. 
* **compute_error_statistics_for_tables.py** is used for computing average performance metrics based on the output of the previous script. 

Subdirectories:
* **core** - Code defining the network and the loss functions.
* **data** - Code for data loading, code describing the structure of the graphs used in the project, etc.
* **experiment_configs** - YAML configuration files: each file contains the parameters used for one training run.
* **utils** - Code for performing various auxiliary tasks.

### Training PoserNet
Run the training script by specifying the configuration file for an experiment 
(otherwise the default configuration is used), like this:

    python train.py --config-file experiment_configs/small_graphs_normal_000000.yaml

## Replicating the experiments in the paper
### Training various models of PoserNet 
You can train PoserNet multiple times, using the experiment configurations provided in this repo, by executing **run_experiments.sh**:

    ./run_experiments.sh

### Measuring the performance of PoserNet
You can assess the performance of the models trained at the previous step (and store the corresponding output) by executing **measure_errors_on_input_and_output_data.py**:

    python measure_errors_on_input_and_output_data.py

The output data will be stored in multiple files in: $dataset_path/output_for_stats/.

### Computing the metrics indicated in the paper
Creating the plots and computing the statistics on the effect of PoserNet on relative poses can be done by running: **compute_error_statistics_for_tables.py**. 

    python compute_error_statistics_for_tables.py


## Data Structures

The original images can be found at the official [7Scenes website](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
All paths stored in the JSON structure (e.g. the path to the original 7Scenes images) assume the dataset is extracted in '/data01/datasets/7Scenes/'. 
All paths stored in the JSON files were only used to efficiently recover the original information for debugging purposes,
and are not used in the code. While providing a full description of the data structure, we mark with [square brackets] 
entries that can be omitted.

### JSON file structure
* train/test &nbsp; -> &emsp; There are two separate lists, containing graphs used for training and testing 
  * #graph_id &nbsp; -> &emsp; Numerical ID of the graph in the list
    * cameras &nbsp; -> &emsp; This dictionary contains information on the cameras included in the graph
      * #cam_id_i &nbsp; -> &emsp; ID of the camera, as a string;  
        * camera_to_world &nbsp; -> &emsp; Ground truth camera-to-world extrinsics, as a 4x4 list
        * K &nbsp; -> &emsp; Ground truth camera intrinsics, as a 3x3 list 
        * K_i &nbsp; -> &emsp; Inverse of the round truth camera intrinsics, as a 3x3 list
        * [image_path] &nbsp; -> &emsp; Path to the original image, for reference. 
        * [depth_path] &nbsp; -> &emsp; Path to the original image, for reference. 
        * [pose_path] &nbsp;&nbsp; -> &emsp; Path to the txt file with the extrinsics, for reference. 
        * [boxes_path] &nbsp; -> &emsp; Path to the object detection files, for reference.
        * matches &nbsp; &nbsp; -> &emsp; Dictionary of the detection of camera #cam_id_i ...
          * #cam_id_f &nbsp; -> &emsp; ... matched to detection in #cam_id_j
            * bb_i &nbsp; -> &emsp; Bounding box in #cam_id_i, expressed as (upper_left_corner_x, upper_left_corner_y, lower_right_corner_x, lower_right_corner_y)
            * bb_f &nbsp; -> &emsp; Bounding box in #cam_id_i, expressed as (upper_left_corner_x, upper_left_corner_y, lower_right_corner_x, lower_right_corner_y)
    * detections  &nbsp; -> &emsp; This dictionary provides information on the object detections and their matches
      * detection_bb &nbsp; -> &emsp; List of all detections bounding boxes, expressed as (upper_left_corner_x, upper_left_corner_y, lower_right_corner_x, lower_right_corner_y)
      * box_matches_sparse &nbsp; -> &emsp; Sparse matrix of how the elements of *detection_bb* are connected, expressed as a list of the index pairs corresponding to matched detections
      * detection_cam  &nbsp; -> &emsp; List containing the id #cam_id_i associated with each detection in *detection_bb*
      * detection_seq &nbsp; -> &emsp; List containing the id #cam_id_i associated with each detection in *detection_bb*
      * covisibility &nbsp; -> &emsp; Square matrix of size n_cams x n_cams with value (i,j) = 1 if the i-th and j-th *cameras* entries are connected through at least five detections, and 0 otherwise 
    * relative_poses &nbsp; -> &emsp; Dictionary of the relative transformations between connected cameras (i.e. camera pairs with a value of 1 in *detections/covisibility* )
      * #cam_id_i &nbsp; -> &emsp; ID of the camera from which the relative pose is computed...
        * #cam_id_f &nbsp; -> &emsp; ... and ID of the camera to which the relative pose is computed.
          * ground_truth &nbsp; -> &emsp; Ground truth relative pose, i.e. transformation from the extrinsics of camera #cam_id_i to those of #cam_id_i, computed from the ground-truth extrinsics
          * feature_5pt &nbsp; -> &emsp; Noisy relative pose transformation, computed using OpenCV 5-point algorithm on the SuperGlue keypoints in the detections matched between #cam_id_i and #cam_id_f
          * bb_centre_5pt &nbsp; -> &emsp; Noisy relative pose transformation, computed using OpenCV 5-point algorithm on the centers of the bounding boxes of the detections matched between #cam_id_i and #cam_id_f
          * [free_keypoint] &nbsp; -> &emsp; Noisy relative pose transformation, computed using OpenCV 5-point algorithm on the SuperGlue keypoints between #cam_id_i and #cam_id_f
    * [origin]
      * file  &nbsp; -> &emsp; For efficiency, the graphs were generated in parallel and working on different scenes separately; this is the JSON file in which the graph was originally generated, for reference 
      * id &nbsp; -> &emsp; This is the graph ID in *origin/file*

## Acknowledgements

This code was developed as part of the [MEMEX](https://memexproject.eu/en/) project, and has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 870743.
