#!/usr/bin/env bash

./scripts/build_container.sh

echo "Exporting container to NoduleAnalyzer.tar.gz"

docker save noduleanalyzer:latest | gzip -c --verbose > NoduleAnalyzer.tar.gz
