#!/bin/bash

set -e

export PYTHONPATH=$(pwd)

python wavenet_tf/train.py