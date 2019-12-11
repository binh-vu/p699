"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
import glob
import json
from dataclasses import dataclass

import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm.auto import tqdm

from wavenet_tf import WaveNetModel, optimizer_factory


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent


@dataclass
class TrainParams:
    data_dir: str = str(ROOT_DIR / 'data' / 'fma_small_25_16000')
    log_dir: str = str(ROOT_DIR / "logdir")
    checkpoint_every: int = 1000
    num_steps: int = int(1e5)
    batch_size: int = 1
    sample_size: int = 100000
    learning_rate: float = 1e-4
    max_to_keep: int = 5
    store_metadata: bool = False
    optimizer: str = 'adam'
    epsilon: float = 0.001
    momentum: float = 0.9
    l2_regularization_strength: float = 0.0
    histograms: bool = False

BATCH_SIZE = 1
DATA_DIRECTORY = str(ROOT_DIR / 'data' / 'fma_small_25_16000')
WAVENET_PARAMS = str(ROOT_DIR / 'data' / 'tf_wavenet_params.json')
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False


def get_dataloader(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    sample_size = args.sample_size

    def gen():
        for file in files:
            data = np.load(file)
            audios = data['audios']
            audios = audios[:, :sample_size].reshape(audios.shape[0], -1, 1)
            for i in range(len(audios)):
                yield audios[i]

    approx_examples = len(files) * 100
    return gen, approx_examples


def train(args: TrainParams, model_params):
    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        receptive_field = WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
                                                                 wavenet_params["dilations"],
                                                                 wavenet_params["scalar_input"],
                                                                 wavenet_params["initial_filter_width"])
        generator, approx_n_examples = get_dataloader(args)
        approx_epoch_size = approx_n_examples // args.batch_size
        dataset = tf.data.Dataset.from_generator(generator, tf.float32, (args.sample_size, 1))
        dataset = dataset.repeat().batch(args.batch_size)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        audio_batch = iterator.get_next()

    # Create network.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
        histograms=args.histograms,
        global_condition_channels=None,
        global_condition_cardinality=None)

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    loss = net.loss(input_batch=audio_batch,
                    global_condition_batch=None,
                    l2_regularization_strength=args.l2_regularization_strength)
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)
    trainable = tf.compat.v1.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.compat.v1.summary.FileWriter(logdir)
    writer.add_graph(tf.compat.v1.get_default_graph())
    run_metadata = tf.compat.v1.RunMetadata()
    summaries = tf.compat.v1.summary.merge_all()

    # Set up session
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    sess.run(iterator.initializer)

    # Saver for storing checkpoints of the model.
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    step = None
    last_saved_step = saved_global_step
    try:
        total = args.num_steps - saved_global_step - 1
        pbar = tqdm(
            total=total,
            initial=saved_global_step + 1,
            desc=f'train (epoch-size={approx_epoch_size}, #epoch={total // approx_epoch_size})')

        for step in range(saved_global_step + 1, args.num_steps):
            if args.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                summary, loss_value, _ = sess.run(
                    [summaries, loss, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata,
                                        'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim])
                writer.add_summary(summary, step)

            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss_value, epoch=step // approx_epoch_size)

            if step > 0 and step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args: TrainParams):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.log_dir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.log_dir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.log_dir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


if __name__ == '__main__':
    train()
