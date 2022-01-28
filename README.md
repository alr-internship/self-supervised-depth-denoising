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
- preprocess
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
======

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
        └── roadmap.md     <- Lists the general roadmap
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Train Model on BwUniCluster 2.0
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

- [ ] remove NaNs from RGB and depth images
    All NaN values get replaced with 0
    Idea: Add mask to tell net where mask values are located

- [ ] normalize dataset
- [ ] mask data
- [X] apply augmentation to dataset
    Applies augmentation to dataset, when the enable_augmentation flag is active.
    Augmentations are chosen randomly from a set of common ones.
    Package used is imgaug