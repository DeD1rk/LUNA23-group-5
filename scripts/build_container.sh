#!/usr/bin/env bash


#Check if Dockerfile is in the current directory
if [ ! -f Dockerfile ]; then
    echo "Dockerfile not found, please run this script from the root of the project directory"
    exit 1
fi

#docker build -t noduleanalyzer "$SCRIPTPATH"
docker buildx build -t noduleanalyzer:latest . 