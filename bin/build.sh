#!/usr/bin/env bash
cd "$(dirname "$0")" && cd ..
bash bin/clean.sh
export ENABLE_SETUP_LONG_DESCRIPTION="TRUE"
python setup.py sdist bdist_wheel -v
