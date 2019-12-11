import librosa, glob, argparse, os, numpy as np
from tqdm.auto import tqdm
from config import *
from typing import *
from multiprocessing.pool import Pool


def create_audio_data(audio_files: List[str], sample_rate: int, outfile: str, mono=True, max_seconds: Optional[int]=None, show_progress: bool=True):
    if max_seconds is not None:
        max_seconds = sample_rate * max_seconds
    
    length = -1
    records = []
    if show_progress:
        iter_audio_files = tqdm(audio_files)
    else:
        iter_audio_files = iter(audio_files)

    for audio_file in iter_audio_files:
        try:
            record, _ = librosa.load(audio_file, sr=sample_rate, mono=mono)
        except:
            print(f"\tInvalid file: {audio_file}")
            continue
    
        record = record[:max_seconds]
        if length == -1:
            length = record.shape[0]
        else:
            # ignore audios less than the length
            if length != record.shape[0]:
                print(f"\tIgnore file due to length is too short {record.shape[0]} (should be {length}). File: {audio_file}")
                continue
                # raise Exception(f"Audio should be in the same shape for fast processing. Try to change the max_seconds. Current length: {record.shape[0]}, previous length: {length}")
        records.append(record)

    records = np.stack(records, axis=0)
    print(">>> finished loading all audios and create an array of", records.shape)
    np.savez_compressed(outfile, audios=records, audio_files=audio_files)


def wrapped_create_audio_data(args):
    return create_audio_data(*args)


def parallel_create_audio_data(audio_files: List[str], sample_rate: int, outfile: str, mono=True, max_seconds: Optional[int]=None):
    fn_args = []
    batch_size = 100

    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        if len(batch) == 0:
            continue
        
        batch_outfile = str(Path(outfile).parent / f"{Path(outfile).stem}.chunk.{i:05}.npz")
        if os.path.exists(batch_outfile):
            continue

        fn_args.append((
            batch,
            sample_rate,
            batch_outfile,
            mono,
            max_seconds,
            False
        ))
    
    pool = Pool()
    for _ in tqdm(pool.imap_unordered(wrapped_create_audio_data, fn_args), total=len(fn_args)):
        pass


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
    parser.add_argument("--parallel",
                        "-p",
                        action='store_true',
                        default=False,
                        help='run in parallel')
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

    if args.parallel:
        parallel_create_audio_data(audio_files, args.sample_rate, args.output_file, max_seconds=args.max_seconds)
    else:
        create_audio_data(audio_files, args.sample_rate, args.output_file, max_seconds=args.max_seconds)