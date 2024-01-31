# Deep learning in practice (DLIP, MVA 2023-2024) practical session framework

Adapted template from https://github.com/MashedP/dlip-pytorch-DL-good-practises.

This framework uses hydra and tensorboard to monitor deep learning experiments. The current code is designed to train simple linear models on the USPS dataset.

## Project Structure

```
├── LICENSE            <- Information about the license of the code. [See more here](https://choosealicense.com/)
├── README.md          <-  A README file to understand the project setup and instructions.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Graphics and figures for use in reports.
│ 
├── requirements.txt   <- Required libraries and dependencies. 
│
├── pyproject.toml     <- Make the project pip installable with `pip install -e`.
├── src/dlip           <- Source code of the project. `dlip` is the name of the package, it can be imported using `import dlip`
│   ├── __init__.py    <- Initializes the 'dlip' Python package.
│   │
│   ├── main.py        <- Main script to train linear models on USPS with a yaml config file.
│   │
│   ├── conf           <- Configuration files for experiments (YAML files managed by Hydra).
│   │   └── train_linear.yaml
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── data.py    <- Load the dataset and split it randomly into train and validation sets.
│   │   └── usps.py    <- Download the USPS dataset.
│   │
│   ├── models         <- Scripts to train models, and make predictions
│   │   ├── evaluation.py
│   │   ├── models.py
│   │   └── train_model.py
│   │
│   ├── utils         <- Useful scripts
│   │   ├── logger.py <- Custom SummaryWritter to log into tensorboard
│   │   └── utils.py  <- Useful functions to instanciate class from yaml config file
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
|   |
│   └── experiment_results  <- Centralize the output and multirun folders created by Hydra


```

This structure is originally inspired by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

## Good Practices

### Code Packaging 

In python, the best way to load a module is to package it and install it.  There are several library for packaging code. Here we use [SetupTools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

`requirements.txt` is an important file for reproducibility. It contains all packages required to launch your experiment. 

To package the project,  go to the root of the project and launch

```pip install -r requirements.txt  install -e . ```

### Seeds 

Using *seeds* is important to assess reproducibility of the results. Learn how to manage randomness in PyTorch [here](https://pytorch.org/docs/stable/notes/randomness.html).

### Argument Parsing with Hydra

Project parameters can be efficiently managed with [Hydra](https://hydra.cc/docs/intro/), greatly simplifying the process of running experiments with various configurations.

Hydra allows you to parse arguments from the command line, to launch and log multiple experiments with different configuration easily. 

To run the training script, you can for example run 

```python src/dlip/models/train_model.py```

or with a different batch size :

```python src/dlip/models/train_model.py train.batch_size=20```

or launching multiple experiment on various batch sizes :

```python src/dlip/models/train_model.py --multirun train.batch_size=10,20,30,40,50```

It helps when you have to perform sweeps on hyperparameters. 

Default Configurations and values of hyperparameters such as batch-size, learning rate, optimizers are stored in a yaml file ```src/dlip/conf/train_model.yaml```

You can access the configuration & outputs of previously run scripts by default in the ```experiment_results/outputs``` folder.


### Experiment Logging with Tensorboard

For tracking experiments, tensorboard is used as it offers a simple interface and can easily be customized. The tfevents files are automatically stored in the hydra subfolders which allows to explore the results by calling `tensorboard --logdir=.` from a parent folder.
