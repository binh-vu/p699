#!/bin/bash

set -e

#python -W ignore audio2npz.py -i '../data/fma_small/*/*.mp3' -r 16000 -o ../data/fma_small_16k_15s/raw/audios.npz -p -s 15
#python -W ignore audio_postprocessing.py -f 'create_metadata' -i '../data/fma_small_16k_15s/raw/*.npz' -o '../data/fma_small_16k_15s/metadata.json'
python -W ignore audio_postprocessing.py -f 'create_logmelspec' -i '../data/fma_small_16k_15s/raw/*.npz' -o '../data/fma_small_16k_15s/logmelspec' -r 16000 -w 64