import numpy as np, glob, ujson
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
print(tf.__version__)
import wavenet_tf as tf_wn

with open("./data/tf_wavenet_params.json", "r") as f:
    wavenet_params = ujson.load(f)

coord = tf.train.Coordinator()
reader = tf_wn.AudioReader(
    './data/fma_small/100',
    coord,
    sample_rate=wavenet_params['sample_rate'],
    gc_enabled=None,
    receptive_field=tf_wn.WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
                                                           wavenet_params["dilations"],
                                                           wavenet_params["scalar_input"],
                                                           wavenet_params["initial_filter_width"]),
    sample_size=100000,
    silence_threshold=None)


audio_batch = reader.dequeue(32)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

threads = tf.train.start_queue_runners(sess=sess, coord=coord)
reader.start_threads(sess)

print(">>> load variable", flush=True)
x = sess.run(audio_batch)

print(x.shape)
print(">>>done")