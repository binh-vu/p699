from typing import List

import numpy as np, glob, torch
from tqdm.auto import tqdm


def load_data(quantized_audio_files: List[str], spect_files: List[str], n_split=1, max_time_steps=None):
    xspects = []
    xquants = []

    for quant_file, spect_file in zip(quantized_audio_files, spect_files):
        xspect = np.load(spect_file)['value']
        xquant = np.load(quant_file)['value']

        if n_split > 1 and max_time_steps is None:
            assert xspect.shape[0] % n_split == 0, f"{xspect.shape[0]} % {n_split} != 0"
            spect_chunk_size = xspect.shape[0] // n_split
            quant_chunk_size = xquant.shape[0] // n_split

            for i in range(n_split):
                xspect_chunk = xspect[i * spect_chunk_size:(i + 1) * spect_chunk_size]
                xquant_chunk = xquant[i * quant_chunk_size:(i + 1) * quant_chunk_size]

                xspects.append(xspect_chunk)
                xquants.append(xquant_chunk)
        elif max_time_steps is not None:
            assert xquant.shape[0] % max_time_steps == 0, f"{xquant.shape[0]} % {max_time_steps} != 0"
            n_reduce = xquant.shape[0] // max_time_steps
            assert xspect.shape[0] % n_reduce == 0, f"{xspect.shape[0]} % {n_reduce} != 0"
            xspects.append(xspect[:xspect.shape[0] // n_reduce])
            xquants.append(xquant[:max_time_steps])
        else:
            xspects.append(xspect)
            xquants.append(xquant)

        # if len(xquants) > 5:
        #     break

    xspects = np.stack(xspects, axis=0)
    xquants = np.stack(xquants, axis=0)

    # local condition would be xspects
    # convert to torch and make it as BatchSize x Channel x Times
    xspects = torch.transpose(torch.from_numpy(xspects), 2, 1)
    # we still keep it as BatchSize x Times
    xquants = torch.from_numpy(xquants)

    print(f"c_batch={xspects.shape} x={xquants.shape}")
    # not converting xquants to one hot encoder because it would be too big
    return xquants, xspects


def get_data_loader(quantized_audio_files: List[str], spect_files: List[str], batch_size, shuffle: bool, pin_memory=True, local_conditioning=False, n_split=1, max_time_steps: int=None):
    x, c = load_data(quantized_audio_files, spect_files, n_split, max_time_steps)
    if local_conditioning:
        args = (x, c)
    else:
        args = (x,)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*args),
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=pin_memory)


if __name__ == "__main__":
    load_data(sorted(glob.glob("../data/fma_tiny*/processed/*.npz")), 2)
