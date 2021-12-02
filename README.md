Self-Supervices Depth-Denoising
==============================

The Project structure:
- [Setup](#setup)
  - [Get Resources](#get-resources)
  - [Create Environment](#create-environment)
  - [Update Environment](#update-environment)
- [Project Organization](#project-organization)

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
To clone them, dvc **and dvc-ssh* must be installed on the system.
Visit [https://dvc.org/doc/install](https://dvc.org/doc/install) for further information.

Afterwards the resources can be pulled with following command

    dvc pull

**Note: The used dvc remote server isn't available to the public.**

Create Environment
-----

To create a specific conda environment,
there is a .yml file located in [envs](./envs), that should be used.
All files are named *depth-denoising_%STAGE%.yml*,
where *%STAGE%* must be replaced with one of the stages listed above.
To ensure all notebooks execute correctly, the last command adds the PYTHONPATH 
environment variable to the environment pointing to the src folder.

To create an environment execute following commands 

    conda env create -f ./envs/depth-denoising_%STAGE%.yml 
    conda activate depth-denoising_%STAGE%
    conda env config vars set PYTHONPATH=$PWD/src

E.g. to generate the environment for training, execute:

    conda env create -f ./envs/depth-denoising_training.yml
    conda activate depth-denoising_training
    conda env config vars set PYTHONPATH=$PWD/src

**Note: To create the environment of stage dataset, the [Zivid SDK v2.5.0](https://www.zivid.com/downloads) must be installed first. Otherwise the above commands will fail.**

Update Environment
------------------
Make sure you are in the correct conda environment
    conda activate depth-denoising_%STAGE%

execute following commands to update conda and pip files

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


