mkdir -p ./ckpts/wavenet
PYTHONPATH=$(pwd) python scripts/train_wavenet.py -m ./data/fma_small_8k/metadata.json -i ./data/fma_small_8k/processed_6s
