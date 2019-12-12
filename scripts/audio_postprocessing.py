import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np
from typing import *

import ujson
from tqdm.auto import tqdm

from config import DATA_DIR


def create_metadata(npz_files: List[str], outfile: str):
    indexes = {}
    for npz_file in tqdm(npz_files):
        data = np.load(npz_file, mmap_mode='r')
        assert data['audios'].shape[0] == data['audio_files'].shape[0]

        for audio_file in data['audio_files']:
            assert audio_file not in indexes
            indexes[audio_file] = npz_file

    with open(outfile, "w") as f:
        ujson.dump(indexes, f)


def create_logmelspec(npz_files: List[str], output_dir: str, sample_rate: int, window_length_ms: int):
    # 1024 if sample rate is 16k and window_length is 64ms
    n_fft = int(window_length_ms * sample_rate / 1000)
    for npz_file in tqdm(npz_files):
        data = np.load(npz_file, mmap_mode='r')
        spects = []
        for audio in tqdm(data['audios']):
            # hop-length is going to be window_size / 4 (vocader set window_length_ms to be 64ms)
            spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4)
            spect = librosa.power_to_db(spect, ref=np.max)
            # (hz, time)
            spects.append(spect)

        np.savez_compressed(os.path.join(output_dir, Path(npz_file).name), spects=spects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_files",
                        "-i",
                        help="glob that find all npz files")
    parser.add_argument("--sample_rate",
                        "-r",
                        type=int,
                        default=16000,
                        help="output dir")
    parser.add_argument('--fft_size',
                        '-w',
                        type=int,
                        help='window size for fourier transform in ms')
    parser.add_argument("--output",
                        "-o",
                        help='output file or directory depends on function')
    parser.add_argument("--parallel",
                        "-p",
                        action='store_true',
                        default=False,
                        help='run in parallel')
    parser.add_argument("--func",
                        "-f",
                        help='function to run')

    args = parser.parse_args()

    if args.npz_files.startswith("/"):
        npz_files = args.npz_files
    else:
        npz_files = str(DATA_DIR / args.npz_files)
    npz_files = sorted(glob.glob(npz_files))

    if args.func == 'create_metadata':
        create_metadata(npz_files, args.output)
    elif args.func == 'create_logmelspec':
        create_logmelspec(npz_files, args.output, args.sample_rate, args.fft_size)
    else:
        raise Exception("Invalid function: %s" % args.func)