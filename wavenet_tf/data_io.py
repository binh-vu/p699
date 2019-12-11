import glob, numpy as np, tensorflow as tf


def get_train_dataset(pattern: str, sample_size: int):
    files = sorted(glob.glob(pattern))
    n_examples = 0

    print(">>> get n_examples", flush=True)
    for file in files:
        data = np.load(file, mmap_mode='r')
        n_examples += data['audio_files'].shape[0]
    print(">>> finish", n_examples, flush=True)

    def generator():
        for file in files:
            data = np.load(file)
            audios = data['audios']
            audios = audios[:, :sample_size].reshape(audios.shape[0], -1, 1)
            for i in range(len(audios)):
                yield audios[i]

    dataset = tf.data.Dataset.from_generator(generator, tf.float32, (sample_size, 1))
    return dataset, n_examples
