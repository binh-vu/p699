#!/bin/bash

set -e

trim_sec=6
max_sec=15
sr=8
sample_rate="${sr}000"
dname="fma_small_${sr}k"
win_len=64

# echo convert mp3 to npz
# python -W ignore audio2npz.py -i '../data/fma_small/*/*.mp3' -r $sample_rate -o "../data/$dname/raw_${max_sec}s/audios.npz" -p -s $max_sec

#echo create metadata of the dataset
#python -W ignore audio_postprocessing.py -f 'create_metadata' \
#  -i "../data/$dname/raw_${max_sec}s/*.npz" \
#  --track_file "../data/fma_metadata/tracks.csv" \
#  --output_metadata "../data/$dname/metadata.json" \
#  --output_dir "../data/$dname/processed_${trim_sec}s"

echo post process the audio
python -W ignore audio_postprocessing.py -f 'process_audios' \
  -i "../data/$dname/raw_${max_sec}s/*.npz" \
  --output_metadata "../data/$dname/metadata.json" \
  --output_dir "../data/$dname/processed_${trim_sec}s" \
  --max_seconds $trim_sec \
  -r $sample_rate -w $win_len