from typing import List

import numpy as np, glob, torch
from tqdm.auto import tqdm


def load_data(npz_files: List[str], n_split=1):
    xspects = []
    xquants = []
    
    for file in npz_files:
        data = np.load(file)

        xspect = data['spects']
        xquant = data['quantized_audios']

        if n_split > 1:
            assert xspect.shape[0] % n_split == 0
            spect_chunk_size = xspect.shape[1] // n_split
            quant_chunk_size = xquant.shape[1] // n_split

            for i in range(n_split):
                xspect_chunk = xspect[:, i*spect_chunk_size:(i+1)*spect_chunk_size]
                xquant_chunk = xquant[:, i*quant_chunk_size:(i+1)*quant_chunk_size]

                xspects.append(xspect_chunk)
                xquants.append(xquant_chunk)
        else:
            xspects.append(xspect)
            xquants.append(xquant)

        break

    xspects = np.concatenate(xspects, axis=0)
    xquants = np.concatenate(xquants, axis=0)

    # local condition would be xspects
    # convert to torch and make it as BatchSize x Channel x Times
    xspects = torch.from_numpy(xspects).permute((0, 2, 1))
    # we still keep it as BatchSize x Times
    xquants = torch.from_numpy(xquants)

    print(f"c_batch={xspects.shape} x={xquants.shape}")
    # not converting xquants to one hot encoder because it would be too big
    return xquants, xspects


def get_data_loader(npz_files: str, batch_size, shuffle: bool, pin_memory=True, n_split=1):
    x, c = load_data(npz_files, n_split)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, c),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    

if __name__ == "__main__":
    load_data(sorted(glob.glob("../data/fma_tiny*/processed/*.npz")), 2)
