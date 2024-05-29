#!/usr/bin/env bash

./scripts/build_container.sh

docker save noduleanalyzer | gzip -c > NoduleAnalyzer.tar.gz
