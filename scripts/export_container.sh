#!/usr/bin/env bash

./scripts/build.sh

docker save noduleanalyzer | gzip -c > NoduleAnalyzer.tar.gz
