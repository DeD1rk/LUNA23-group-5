# LUNA23 group 5

## Project description
The goal of this project is to develop a deep learning model that can do three tasks: segmentation, classification and malignancy prediction of pulmonary nodules. The dataset is provided by the LUNA23 challenge, the test set consists of 687 nodules in the training set and 256 nodules in the testing set. Each nodule in the training set has an associated coordinate, diameter, segmentation, nodule type, and malignancy. The test set only contains the images themselves.

## Sctructure of the project
The project is structured as follows:
 - 📁 `dataset`
    | - 📁 `train_set` 
    | - 📁 `test_set`
    └ - 📄 `luna23-ismi-train-set` contains labels and metadata
 - 📁 `luna` contains all code related to the model
    | - 📄 `__main__.py` provides a CLI to train the model
    | - 📄 `model.py` contains the model
    | - 📄 `training.py` contains the training loop
    | - 📄 `dataset.py` provides interface to the training data
    | - 📄 `utils.py` utility functions 
    └ - 📄 `constants.py` constant settings and facts about input data
 - 📁 `results/<date>_<time>_<exp-id>_fold<fold>/...` outputs from training


## Usage

The code of the project is structured in a python module `luna` that provides a convenient
CLI, allowing to train and perform inference with our model, and select various hyperparameters
without making changes in code. 

After installing dependencies with `pip install -r requirements.txt`,
the CLI can be used as follows:

```
# View all training options:
python -m luna train --help

# Minimal way to train with defaults for everything:
python -m luna train --data-dir=dataset --results-dir=results

# A more complete example, that sets some options.
# This will run inference on the test set as well after training.
python -u -m luna train \
    --data-dir=$TMPDIR/dataset \
    --results-dir=$TMPDIR/results \
    --batch-size=16 \
    --epochs=800 \
    --fold=0 \
    --dropout=0.8 \
    --aug-mirror-x \
    --exp-id="try-very-high-dropout-and-mirror-on-the-x-axis" \
    --perform-inference
```

### Container

To create a new algorithm container the latest checkpoint of the model must be placed in `checkpoints/best-model` folder.
We can build the container with the following command:

```bash
./scripts/build_container.sh
```
To test if the container is working properly, we can run the following command:

```bash
./scripts/test_container.sh
```


