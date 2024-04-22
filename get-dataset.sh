#!/bin/bash
# This script downloads the dataset for the project

# Check if dependencies are installed, wget and Unzip
if ! [ -x "$(command -v wget)" ]; then
    echo 'Error: wget is not installed.' >&2
    exit 1
fi

if ! [ -x "$(command -v unzip)" ]; then
    echo 'Error: unzip is not installed.' >&2
    exit 1
fi


# Check if the dataset is already present
if [ -d "dataset" ]; then
    echo "Dataset is already present. Exiting the script."
    exit 0
fi

# Download the dataset
wget https://surfdrive.surf.nl/files/index.php/s/5hlX6Nqptik378t/download -O dataset.zip

# Check if the download was successful
# If the download was not successful, exit the script
if [ $? -ne 0 ]; then
    echo "Download failed. Exiting the script."
    exit 1
fi
# If the download was successful, unzip the dataset
# Unzip the dataset
unzip dataset.zip -d data

# Check if the unzip was successful 
if [ $? -ne 0 ]; then
    echo "Unzip failed. Exiting the script."
    exit 1
fi

# Remove the zip file
rm dataset.zip

