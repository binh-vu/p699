import librosa, glob, argparse, os, numpy as np
from tqdm.auto import tqdm
from config import *
from typing import *


def create_audio_data(audio_files: List[str], sample_rate: int, outfile: str, mono=True, max_seconds: Optional[int]=None):
    if max_seconds is not None:
        max_seconds = sample_rate * max_seconds
    
    length = -1
    records = []
    for audio_file in tqdm(audio_files):
        record, _ = librosa.load(audio_file, sr=sample_rate, mono=mono)
        record = record[:max_seconds]
        if length == -1:
            length = record.shape[0]
        else:
            assert length == record.shape[0], "Audio should be in the same shape for fast processing. Try to change the max_seconds"
        records.append(record)

    records = np.stack(records, axis=0)
    print(">>> finished loading all audios and create an array of", records.shape)
    np.savez_compressed(outfile, audios=records, audio_files=audio_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_files",
                        "-i",
                        default="fma-small/*/*.mp3",
                        help="glob that find all audio files")
    parser.add_argument("--sample_rate",
                        "-r",
                        type=int,
                        default=16000,
                        help="output dir")
    parser.add_argument("--output_file",
                        "-o",
                        help='output file')
    parser.add_argument("--max_seconds",
                        "-s",
                        default=None,
                        type=int,
                        help='maximum number of seconds to keep')

    args = parser.parse_args()

    if args.audio_files.startswith("/"):
        audio_files = args.audio_files
    else:
        audio_files = str(DATA_DIR / args.audio_files)
    audio_files = sorted(glob.glob(audio_files))

    create_audio_data(audio_files, args.sample_rate, args.output_file, max_seconds=args.max_seconds)