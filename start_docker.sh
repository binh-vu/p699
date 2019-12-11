#!/bin/bash

set -e

docker run \
  --rm --gpus all -w /ws \
  -v $(pwd):/ws -p 8888:8888 \
  -e PYTHONPATH=/ws/scripts:/ws \
   -it p699 bash
 #  -u $(id -u ${USER}):$(id -g ${USER}) \
#   jupyter lab --ip=0.0.0.0 --allow-root