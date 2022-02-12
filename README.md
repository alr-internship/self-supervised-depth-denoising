Self-Supervised Depth-Denoising
==============================

The Project structure:
- [Setup](#setup)
  - [Get Resources](#get-resources)
  - [Create Environment](#create-environment)
  - [Update Environment](#update-environment)
- [Project Organization](#project-organization)
- [Train Model on BwUniCluster 2.0] (#train-model-on-bwunicluster-2.0)

Setup
=====

The project consists of multiple stages, named:
- dataset
- training
- ...

All those stages have their own [conda](https://anaconda.org) environment,
to reduce the size of each environment and prevent dependency conflicts
(especially with different python versions).

Get Resources
-----

All resources of this repository are stored in [dvc](https://dvc.org).
Therefore, they won't be downloaded when cloning the repository.
To clone them, dvc and dvc-ssh must be installed on the system.
Visit [https://dvc.org/doc/install](https://dvc.org/doc/install) for further information.

Afterwards the resources can be pulled with following command

    dvc pull

**Note: The used dvc remote server isn't available to the public.**

Create Environment
-----

Before creating an environment, reasure the [Zivid SDK v2.5.0](https://www.zivid.com/downloads) is installed on your system.
To create a specific conda environment,
there are .yml files located in [envs](./envs), that should be used.
All files are named *depth-denoising_%STAGE%.yml*,
where *%STAGE%* must be replaced with one of the stages listed above.
To ensure all notebooks execute correctly, the last command given below adds the PYTHONPATH 
environment variable pointing to the src folder to the active conda environment.

To create an environment execute following commands 

    conda env create -f ./envs/depth-denoising_%STAGE%.yml 
    conda activate depth-denoising_%STAGE%
    conda env config vars set PYTHONPATH=$PWD/src

E.g. to generate the environment for *training*, execute:

    conda env create -f ./envs/depth-denoising_training.yml
    conda activate depth-denoising_training
    conda env config vars set PYTHONPATH=$PWD/src

Update Environment
------------------

    conda env update --file ./envs/depth-denoising_%STAGE%.yml

Export Environment
------------------
Make sure you are in the correct conda environment

    conda activate depth-denoising_%STAGE%

execute following commands to update conda and pip packages 

    conda env export > ./envs/depth-denoising_%STAGE%.yml


Project Organization
====

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │   └── roadmap.md     <- Lists the general roadmap
    ├── envs               <- Conda environments
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── resources          <- Trained and serialized models, all datasets
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── _old           <- old code
    │   ├── calibrate      <- files to calibrate frames
    │   │   └── calib_in_pcd.py
    │   ├── camera_interfaces <- interfaces to talk to cameras
    │   ├── dataset        <- Dataset management files
    │   ├── evaluate       <- Scripts to evaluate trained models
    │   ├── models         <- UNet model classes
    │   ├── networks       <- UNet and LSTMUNet model classes
    │   ├── segmentation   <- files to segment YCB dataset
    │   ├── trainers       <- Out-Of-Fold and Basic trainer to train models
    │   ├── utils          <- utils


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Dataset Generation
===

The dataset generation can be split into following steps
- Retrieve Images for Calibration
    - CharuCo
    - 
- 

Train Depth-Denoising Model
===

The main script to train the denoising model is [train_models.py](src/trainers/train_models.py).
To run the script execute
```bash
python train_models.py %CONFIG_PATH%
```
The training can be configured via a configuration script.
A demo config file is located in [src/trainers/config.yml](src/trainers/config.yml).
There are two trainers present to train a model.
- Out-Of-Fold Trainer
- Basic Trainer
Each trainer can be activated/deactivated in the configuration file.
The configuration file contains many arguments that are explained in the config.yml referenced before.
The argument %CONFIG_PATH% must be replaced with the path to the configuration file
that should be used for training.


BwUniCluster 2.0
====

There are multiple scripts located in models, that might be helpful to train all the models on the BwUniCluster 2.0.
To enqueue the training step execute following command in the 

    sbatch ./models/enqueue_job.sh

To test the training, one can execute following command to queue the training on a dev node

    sbatch ./models/enqueue_dev_job.sh

The process will be logged into `train_models.out`

For further information, visit [https://wiki.bwhpc.de/e/BwUniCluster_2.0_Slurm_common_Features](https://wiki.bwhpc.de/e/BwUniCluster_2.0_Slurm_common_Features)


TODOs
=====

- [X] dataset calibration
    Calibrated dataset by map to 3D, apply homogeneous transformation, map to 2D.
    Crop then to overlapping image region.
    This process also introduces NaN values!
- [X] notebook to visualize inference of trained models
    Added new notebook that loads a given model and inferes a given dataset.
    The results get visualized
- [X] remove NaNs from RGB and depth images
    All NaN values get replaced with 0
    Idea: Add mask to tell net where mask values are located
    Problem: Mask only for rs nans or also for zv nans? -> Only for rs nans, as zv NaNs aren't avaliable for test time
    Loss: The networks loss function sums up pixel loss, counting only the ones where no NaN is present in the data
- [X] normalize dataset
    RGB data will be normalized to [0, 1],
    Depth data will not be normalized
- [X] mask data
    Methods:
        - Plane segmentation
            Problem: rs plane is not even enough, parts of object get segmented as plane
        - Region Growing
            Problem: hard to pick correct hyperparameters and thresholds (like color and adjance distance)
        - **Mask-RCNN trained on YCB Video Dataset**
- [X] apply augmentation to dataset
    Applies augmentation to dataset, when the enable_augmentation flag is active.
    Augmentations are chosen randomly from a set of common ones.
    Package used is imgaug
- [ ] Net can output negative values
- [ ] update net dtypes
    Currently float is chosen for all input channels,
    it would make sense to make atleast the mask uint 
- [X] train mask rcnn on new dataset (separate test set)
- [X] evaluate mask rcnn on ycb with separate test set
- [X] JSON for test,train,val set separation
    DatasetInterface should be able to work with that
- [X] add eval code for UNet
- [ ] train and evaluate UNet on
    All nets should have augmentation deactivated
    - [ ] calibrated, not cropped
    - [ ] calibrated, cropped
    - [ ] masked, not cropped
    - [ ] masked, cropped
- [ ] augmented dataset contains 0 for NaNs in out of region