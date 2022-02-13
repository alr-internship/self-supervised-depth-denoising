Self-Supervised Depth-Denoising
==============================

The Project structure:
- [Setup](#setup)
  - [Get Resources](#get-resources)
  - [Create Environment](#create-environment)
  - [Update Environment](#update-environment)
- [Project Organization](#project-organization)
- [BwUniCluster 2.0](#bwunicluster-2.0)

Environment
=====

The project consists of two stages, namely:
- dataset
- training

Those stages have their own [conda](https://anaconda.org) environment,
to reduce the size of each environment and prevent dependency conflicts
(especially with different python versions).

Create Environment
-----

To create the dataset environment, reasure the [Zivid SDK v2.5.0](https://www.zivid.com/downloads) is installed on your system. 
Otherwise the creation might fail.

To create a specific conda environment,
there are .yml files located in [envs](./envs), that should be used.
All files are named *depth-denoising_%STAGE%.yml*,
where *%STAGE%* must be replaced with one of the stages listed above.
To ensure all notebooks execute correctly, the last command given below adds the PYTHONPATH 
environment variable pointing to the src folder of the repository.

To create an environment execute following commands 
```bash
conda env create -f ./envs/depth-denoising_%STAGE%.yml 
conda activate depth-denoising_%STAGE%
conda env config vars set PYTHONPATH=$PWD/src
```

E.g. to generate the environment for *training*, execute:
```bash
conda env create -f ./envs/depth-denoising_training.yml
conda activate depth-denoising_training
conda env config vars set PYTHONPATH=$PWD/src
```

Update Environment
------------------

To update the created environment with the respective environment YAML file, execute:
```bash
conda env update --file ./envs/depth-denoising_%STAGE%.yml
```

Export Environment
------------------

After changes are made to one of the environments, 
ensure to also update the respective YAML file.

Make sure you are in the correct conda environment
```bash
conda activate depth-denoising_%STAGE%
```

Execute following command to export currently used conda and pip packages 
```bash
conda env export > ./envs/depth-denoising_%STAGE%.yml
```

Resources
===

All resources of this repository are stored in [dvc](https://dvc.org).
Therefore, they won't be downloaded when cloning the repository.
To clone them, dvc and dvc-ssh must be installed on the system.
Visit [https://dvc.org/doc/install](https://dvc.org/doc/install) for further information.
They might be already installed in one of the conda environments.

When dvc and dvc-ssh is installed, the resources can be pulled with following command
```bash
dvc pull
```

**Note: The used dvc remote server isn't available to the public.**

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


Dataset Generation
===

The dataset generation includes multiple steps to retrieve a good enough result.
All steps should be executed in the `depth-denoising_dataset` conda environment

1. Retrieve Images for Calibration
1. Compute Extrinsics Calibration
1. Retrieve Dataset
1. Calibrate Dataset
1. Compute Masks for Dataset
1. Augment Dataset
1. Generate JSONs to split Dataset into train/val/test set


Retrieve Images for Calibration
---

At first, the cameras must be calibrated to each other.
Therefore, multiple frames must be recorded, with the help of which
the calibration can be computed.
The kind of frames depends on the calibration method.
- For CharuCo Board calibration, capture frames of a CharuCo Board.
- For Manual Feature Picking, capture frames where features can later be picked manually 
rather easy.

For the capture process, 
use the notebook [00_retrieve_datset.ipynb](notebooks/00_retrieve_dataset.ipynb).
This notebook helps immensively to generate pairs of LQ- and HQ rgb and depth frames.

Compute Extrinsics Calibration
---

There exist two methods to compute the extrinsics transformation matrix 
that maps the frame of the HQ camera plane onto the LQ camera`s plane.

- CharuCo (**preferred**)
- Manual Feature Picking

Both methods print out the resulting transformation matrix.
The matrix must be saved for later.

### CharuCo
This method has the benefit, that the calibration can be automated fully.
For computing the extrinsic transformation matrix with this method,
the script [charuco_compute_extrinsic_transform.py](src/data_processing/charuco_compute_extrinsic_transform.py) should be used.
```bash
python src/data_processing/charuco_compute_extrinsic_transform.py %DIR_TO_FILES%
```
The placeholder `$DIR_TO_FILES$` must be replaced with the directory the charuco dataset is located in.
The script will print out the final transformation matrix at the end.

### Manual Feature Picking
This method requires a human to select corresponding features in the point clouds
of the cameras images.
Use the script [manual_compute_extrinsic_transform.py](src/data_processing/manual_compute_extrinsic_transform.py) to compute the transformation matrix.
This script will prompt two visualization sequentially.
The user must select corresponding points in both point clouds.
To select a point use the Open3D commands, also listed in the respective documentation.
**The order in which the points are selected matter.**
The transformation matrix will be computed to minimize the square distance of the point pairs selected.
The resulting transformation matrix will be printed to stdout at the end of the script.

Retrieve Dataset
---
Calibrate Dataset
---
This step aligns the rgb and depth images of the HQ and LQ cameras with
the earlier computed extrinsic transformation matrix.
The script [calib_in_pcd.py](src/data_processing/calib_in_pcd.py)
should be used to calibrate a dataset.
At the top of the script (~line 14), the transformation matrix is hardcoded into.
This transformation matrix should be replaced with the one computed in the step before.
The script has multiple parameters.
- *$INPUT_DIR%*: The first, positional parameters are the input directory where the raw files are located
that should be computed.
- *$OUTPUT_DIR%*: The second is the output directory the calibrated files are saved to.
The file structure (including the file naming) of the input directory will
be mirrored to the output directory.
Therefore file `%INPUT_DIR%/DIR_1/file.npz` will later be located in `%OUTPUT_DIR%/DIR_1/file.npz`.
- *--jobs*: number of jobs
- *-cropped*: if the calibrated images should be cropped so that no black pixels are visible,
    or the images should be the same as the original, LQ image view.
- *--debug*: displays visualizations of the point clouds before and after calibration.

Compute Masks for Dataset
---
Augment Dataset
---
Generate JSONs to split Dataset into train/val/test set
---

The resulting dataset can be used to train and evaluate the network by passing the paths to the generated jsons


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