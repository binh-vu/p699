import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np, pandas as pd
from typing import *

import ujson
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from config import DATA_DIR
from nnmnkwii import preprocessing as P


class Postprocessing:

    def __init__(self):
        self.mel_basis = None

    def create_metadata(self, npz_files: List[str], track_metadata: str, outfile: str):
        tracks = pd.read_csv(track_metadata, index_col=0, header=[0, 1])
        keep_cols = [('set', 'split'),
                     ('set', 'subset'), ('track', 'genre_top')]

        df_all = tracks[keep_cols]
        df_all = df_all[df_all[('set', 'subset')] == 'small']
        df_all['track_id'] = df_all.index

        dict_genres = {'Electronic': 1, 'Experimental': 2, 'Folk': 3, 'Hip-Hop': 4,
                       'Instrumental': 5, 'International': 6, 'Pop': 7, 'Rock': 8}

        df_train = df_all[df_all[('set', 'split')] == 'training']
        df_valid = df_all[df_all[('set', 'split')] == 'validation']
        df_test = df_all[df_all[('set', 'split')] == 'test']

        track2genre = {}
        for name, df in [('train', df_train), ('valid', df_valid), ('test', df_test)]:
            track2genre[name] = {}
            for idx, row in df.iterrows():
                track_id = int(row['track_id'])
                genre = str(row[('track', 'genre_top')])
                track2genre[name][f"{track_id:06}.mp3"] = dict_genres[genre]

        indexes = {}
        for npz_file in tqdm(npz_files):
            data = np.load(npz_file, mmap_mode='r')
            assert data['audios'].shape[0] == data['audio_files'].shape[0]
            npz_file_name = Path(npz_file).name
            indexes[npz_file_name] = []
            for audio_file in data['audio_files']:
                indexes[npz_file_name].append(Path(audio_file).name)

        with open(outfile, "w") as f:
            ujson.dump({
                "indexes": indexes,
                "train": track2genre['train'],
                "valid": track2genre['valid'],
                "test": track2genre['test'],
            }, f)

    def process_audio_files(self, npz_files: List[str], metadata_file: str, output_dir: str, max_seconds: int,
                            sample_rate: int,
                            window_length_ms: int):
        for dname in ["spects", "audios", "quantized_audios"]:
            (Path(output_dir) / dname).mkdir(exist_ok=True, parents=True)

        # 1024 if sample rate is 16k and window_length is 64ms
        n_fft = int(window_length_ms * sample_rate / 1000)
        assert n_fft % 4 == 0
        hop_length = n_fft // 4

        if max_seconds is not None:
            max_seconds = max_seconds * sample_rate

        constant = P.mulaw_quantize(0.0, 255)
        processed_npz_files = []

        with open(metadata_file, "r") as f:
            metadata = ujson.load(f)

        pbar = tqdm(total=sum(len(metadata[x]) for x in ['train', 'valid', 'test']), desc='process audios')
        for npz_file in npz_files:
            data = np.load(npz_file, mmap_mode='r')
            spects = []
            padded_audios = []
            mulaw_quantized = []

            for audio in data['audios']:
                pbar.update(1)
                if max_seconds is not None:
                    trim_audio = audio[:max_seconds]
                else:
                    trim_audio = audio

                # create melspectrograpm (time, hz)
                spect = self.logmelspectrogram(trim_audio, sample_rate, n_fft, hop_length).astype(np.float32).T

                # quantize contiguous signal to 256 discrete values
                # make sure it is in -1.0 & 1.0, sometimes it goes outside
                audio = np.clip(audio, -1.0, 1.0)
                quanted_audio = P.mulaw_quantize(audio, 255)

                # adjust time resolution between audio and mel-spectrogram so that
                # we can do convolution upsampling
                length = spect.shape[0] * hop_length
                if max_seconds is not None:
                    # instead of padding, we just use correct length
                    quanted_audio = quanted_audio[:length]
                    padded_audio = audio[:length]
                else:
                    quanted_audio = np.pad(quanted_audio, (0, n_fft), mode='constant', constant_values=constant)
                    quanted_audio = quanted_audio[:length]

                    padded_audio = np.pad(audio, (0, n_fft), mode='constant', constant_values=0.0)
                    padded_audio = padded_audio[:length]

                spects.append(spect)
                mulaw_quantized.append(quanted_audio)
                padded_audios.append(padded_audio)

            processed_npz_files.append((spects, mulaw_quantized, padded_audios))
        pbar.close()
        print(">>> normalize spect")
        x = np.asarray([spect for spects, _, _ in processed_npz_files for spect in spects])
        x = x.reshape(-1, x.shape[-1])
        scaler = StandardScaler()
        scaler.fit(x)

        print(">>> write output")
        for npz_file, processed_npz_file in tqdm(zip(npz_files, processed_npz_files), total=len(npz_files)):
            spects, mulaw_quantized, padded_audios = processed_npz_file
            # change shape to normalize and convert it back
            spects = np.asarray(spects).reshape(-1, spects[0].shape[-1])
            spects = scaler.transform(spects)
            spects = spects.reshape(len(mulaw_quantized), -1, spects.shape[-1])

            npz_file_name = Path(npz_file).name
            for i, song_name in enumerate(metadata['indexes'][npz_file_name]):
                np.savez_compressed(os.path.join(output_dir, "spects", song_name.replace(".mp3", "")), value=spects[i])
                np.savez_compressed(os.path.join(output_dir, "audios", song_name.replace(".mp3", "")), value=padded_audios[i])
                np.savez_compressed(os.path.join(output_dir, "quantized_audios", song_name.replace(".mp3", "")),
                                    value=mulaw_quantized[i])

    def logmelspectrogram(self, y, sample_rate, fft_size, hop_length, pad_mode='reflect'):
        # spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=n_fft // 4)
        # spect = librosa.power_to_db(spect, ref=np.max)

        # copied from wavenet_vocoder code, don't know the different between librosa function with this one
        # hop-length is going to be window_size / 4 (vocader set window_length_ms to be 64ms)
        if self.mel_basis is None:
            self.mel_basis = librosa.filters.mel(sample_rate, fft_size, fmin=125, fmax=7600, n_mels=80)

        D = librosa.stft(y=y, n_fft=fft_size, hop_length=hop_length, win_length=fft_size, window="hann",
                         pad_mode=pad_mode)
        S = np.dot(self.mel_basis, np.abs(D))
        S = np.log10(np.maximum(S, 1e-10))
        return S


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_files",
                        "-i",
                        help="glob that find all npz files")
    parser.add_argument("--track_file",
                        help="track file that contains metadata about the songs")
    parser.add_argument("--sample_rate",
                        "-r",
                        type=int,
                        default=16000,
                        help="output dir")
    parser.add_argument('--fft_size',
                        '-w',
                        type=int,
                        help='window size for fourier transform in ms')
    parser.add_argument("--output_metadata",
                        help='output metadata')
    parser.add_argument("--output_dir",
                        help='output directory for postprocessing files')
    parser.add_argument("--max_seconds",
                        "-s",
                        type=int,
                        default=None,
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

    processor = Postprocessing()

    if args.func == 'create_metadata':
        processor.create_metadata(npz_files, args.track_file, args.output_metadata)
    elif args.func == 'process_audios':
        processor.process_audio_files(npz_files, args.output_metadata, args.output_dir, args.max_seconds,
                                      args.sample_rate, args.fft_size)
    else:
        raise Exception("Invalid function: %s" % args.func)
