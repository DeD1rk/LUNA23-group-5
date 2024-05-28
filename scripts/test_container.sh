#!/usr/bin/env bash

./scripts/build-container.sh

#Change the ownership of the input and output directories allow the container to write to them
chmod 777 -R ./test/input
chmod 777 -R ./test/expected_output

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --gpus='"device=0"' \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v ./test/input/:/input/ \
        -v ./test/expected_output/:/output/ \
        noduleanalyzer:validation
