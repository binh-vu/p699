#!/bin/bash

set -e

dname='fma_small_16k_15s'
#python -W ignore audio2npz.py -i '../data/fma_small/*/*.mp3' -r 16000 -o ../data/fma_small_16k_15s/raw/audios.npz -p -s 15
python -W ignore audio_postprocessing.py -f 'create_metadata' \
  -i "../data/$dname/raw/*.npz" \
  --output_metadata "../data/$dname/metadata.json" \
  --output_dir "../data/$dname/logmelspec"

python -W ignore audio_postprocessing.py -f 'process_audios' \
  -i "../data/$dname/raw/*.npz" \
  --output_metadata "../data/$dname/metadata.json" \
  --output_dir "../data/$dname/processed" \
  --max_seconds 3 \
  -r 16000 -w 64