# Fork specific Overview:
This fork of the original work is intended as a repository for Team 4's project in the class KI in der Produktion. 

# Instructions for starting:
1. Download and extract your chosen dataset to the folder data/

2. Build the dockerfile using the following command:

    &emsp; Option 1
    ```bash
    $ /build_docker.sh
    ```
    
    &emsp; Option 2 (windows -- works (requirements: Docker Desktop installed))
    ```bash
    $ docker compose build
    ```

3. Start the container from the root directory of this repository:
    
    &emsp; Option 1
    ```bash
    $ ./start_docker.sh
    ```

    &emsp; Option 2
    ```bash
    $ docker compose up
    ```
    (better start the container on windows from the Docker Desktop GUI)

4. (Optional) If using the cornell dataset and depth images haven't been created from pcd:

  ```bash
  $ python -m utils.dataset_processing.generate_cornell_depth data/
  ```

5. To train the model:
```bash
$ python train_network.py --dataset <cornell or jacquard> --dataset-path data/ --description training_cornell
```

train the modifies model with rfb block and one residual block in the embedding/ bottleneck like in paper "lightweight cnn with gaussian based grasping representation for robotic grasping detection"

```bash
$ python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_rfb --epochs 10 --description training_cornell_rfb
```
python train_network.py --dataset graspnet --dataset-path data/ --network grconvnet3_rfb --epochs 10 --description training_cornell_rfb
train the modified model with rfb block, multi dimensional fusion and one residual block in the embedding/ bottleneck like in paper "lightweight cnn with gaussian based grasping representation for robotic grasping detection"

```bash
$ python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_rfb_mdaf_single --epochs 10 --description training_cornell_rfb_mdfa_single
```

train the modified model with rfb block, 3 RES-Blocks, 2 multi dimensional fusion and concatenation of shallow and deep features in upsampling (part of "lightweight cnn with gaussian based grasping representation for robotic grasping detection")

```bash
$ python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_rfb_mdaf_multi_lightweight --epochs 10 --description training_cornell_rfb_mdfa_multi_lightweight
```

train the lightweight model with ResBlocks, RFB, MDAF, MaxPool ("lightweight cnn with gaussian based grasping representation for robotic grasping detection")

```bash
$ python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight --epochs 10 --description training_cornell_lightweight
```

6. View logs with tensorboard
```
tensorboard --logdir=logs
```
Then open the following url in your browser: http://localhost:6006/

7. Visualize Training and Evaluation pictures:
set the following flag at the end of your command line when calling the train_network.py or the evaluation.py!

```bash
$ --vis
 ```

----------------------------------------------------------------------------------------------------------------------------------

# Antipodal Robotic Grasping
We present a novel generative residual convolutional neural network based model architecture which detects objects in the camera’s field of view and predicts a suitable antipodal grasp configuration for the objects in the image.

This repository contains the implementation of the Generative Residual Convolutional Neural Network (GR-ConvNet) from the paper:

#### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/antipodal-robotic-grasping-using-generative/robotic-grasping-on-cornell-grasp-dataset)](https://paperswithcode.com/sota/robotic-grasping-on-cornell-grasp-dataset?p=antipodal-robotic-grasping-using-generative)

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:

```
@inproceedings{kumra2020antipodal,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}}
```

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/skumra/robotic-grasping.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd robotic-grasping
$ pip install -r requirements.txt
```

## Datasets

This repository supports both the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

Example for Jacquard dataset:

```bash
python train_network.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard --use-dropout 0 --input-size 300
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

Example for Jacquard dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --iou-eval --use-dropout 0 --input-size 300
```

## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python run_grasp_generator.py
```

## Run on a Robot
To run the grasp generator with a robot, please use our ROS implementation for Baxter robot. It is available at: https://github.com/skumra/baxter-pnp
