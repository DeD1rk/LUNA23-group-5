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

# Handige linkjes :)
1. https://pubs.rsna.org/doi/10.1148/radiol.223308
"Prior CT Improves Deep Learning for Malignancy Risk Estimation of Screening-detected Pulmonary Nodules"

2. https://pubs.rsna.org/doi/epdf/10.1148/radiol.2021204433
"Deep Learning for Malignancy Risk Estimation of Pulmonary Nodules Detected at Low-Dose Screening CT"
They only did malignancy prediction. They used 9 2D slices and an existing 3D model (usually used on videos) to classify the images. 

3. https://www.sciencedirect.com/science/article/pii/S1746809422002233#s0020
"Benign-malignant classification of pulmonary nodule with deep feature optimization framework"
Utilizing a deep feature optimization framework DFOF to enhance bening-malignant classification. This is achieved by spliting the network in 2 streams. One searching for intranodular and perinodular features and the other for revealing intranodular information.

4. https://www.sciencedirect.com/science/article/pii/S0895611121000343#sec0010 " On the performance of lung nodule detection, segmentation and classification" (2021). Gives a review of many algortihms and their accuracies. Small thing about multi task learning and that it is effective. Probs mostly relevant for when we are writing the paper. 


https://github.com/hassony2/kinetics_i3d_pytorch
aformentioned 3D model used in the 2nd paper

Multi task learning papers:

1. https://github.com/uci-cbcl/NoduleNet,https://link.springer.com/chapter/10.1007/978-3-030-32226-7_30 . "First, because of the mismatched goals of localization and classification, it may be sub-optimal if these two tasks are performed using the same feature map. Second, a large receptive field may integrate irrelevant information from other parts of the image, which may negatively affect and confuse the classification of nodules, especially small ones." Made a MTL and describe their choices, code for the network can be found at the github.
2. https://github.com/CaptainWilliam/MTMR-NET, https://ieeexplore.ieee.org/document/8794587 (hiervan lukt t me om een of andere reden niet de paper daadwerkelijk te openen), maar wel code van een MLT i guess
 
