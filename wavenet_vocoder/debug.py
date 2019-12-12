import glob, numpy as np
from warnings import warn

import torch
from torch import optim, nn
from wavenet_vocoder.util import is_mulaw_quantize, is_scalar_input
from wavenet_vocoder import WaveNet
print(torch.__version__)

from wavenet_vocoder.tfcompat.hparam import HParams



hparams = HParams(
    name="wavenet_vocoder",

    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    # **NOTE**: if you change the one of the two parameters below, you need to
    # re-run preprocessing before training.
    input_type="mulaw-quantize",
    quantize_channels=256,  # 65536 or 256

    # Audio:
    # time-domain pre/post-processing
    # e.g., preemphasis/inv_preemphasis
    # ref: LPCNet https://arxiv.org/abs/1810.11846
    preprocess="",
    postprocess="",
    # waveform domain scaling
    global_gain_scale=1.0,

    sample_rate=16000,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    win_length=1024,
    win_length_ms=-1.0,
    window="hann",

    # DC removal
    highpass_cutoff=70.0,

    # Parametric output distribution type for scalar input
    # 1) Logistic or 2) Normal
    output_distribution="Logistic",
    log_scale_min=-16.0,

    # Model:
    # This should equal to `quantize_channels` if mu-law quantize enabled
    # otherwise num_mixture * 3 (pi, mean, log_scale)
    # single mixture case: 2
#     out_channels=10 * 3,
    out_channels=256,
    layers=24,
    stacks=4,
    residual_channels=128,
    gate_channels=256,  # split into 2 gropus internally for gated activation
    skip_out_channels=128,
    dropout=0.0,
    kernel_size=3,

    # Local conditioning (set negative value to disable))
    cin_channels=80,
    cin_pad=0,
    # If True, use transposed convolutions to upsample conditional features,
    # otherwise repeat features to adjust time resolution
    upsample_conditional_features=True,
    upsample_net="ConvInUpsampleNetwork",
    upsample_params={
        "upsample_scales": [4, 4, 4, 4],  # should np.prod(upsample_scales) == hop_size
    },

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker dataset
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=7,  # 7 for CMU ARCTIC

    # Data loader
    pin_memory=True,
    num_workers=1,

    # Loss

    # Training:
    batch_size=8,
    optimizer="Adam",
    optimizer_params={
        "lr": 1e-3,
        "eps": 1e-8,
        "weight_decay": 0.0,
    },

    # see lrschedule.py for available lr_schedule
    lr_schedule="step_learning_rate_decay",
    lr_schedule_kwargs={"anneal_rate": 0.5, "anneal_interval": 200000},

    max_train_steps=1000000,
    nepochs=2000,

    clip_thresh=-1,

    # max time steps can either be specified as sec or steps
    # if both are None, then full audio samples are used in a batch
    max_time_sec=None,
    max_time_steps=10240,  # 256 * 40

    # Hold moving averaged parameters and use them for evaluation
    exponential_moving_average=True,
    # averaged = decay * averaged + (1 - decay) * x
    ema_decay=0.9999,

    # Save
    # per-step intervals
    checkpoint_interval=100000,
    train_eval_interval=100000,
    # per-epoch interval
    test_eval_epoch_interval=50,
    save_optimizer_state=True,

    # Eval:
)

def build_model(hparams):
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
    )
    return model

model = build_model(hparams)
model

audio_files = sorted(glob.glob("../data/fma_tiny_16k_15s/raw/*.npz"))
processed_files = sorted(glob.glob("../data/fma_tiny_16k_15s/processed/*.npz"))

data = np.load(processed_files[0])
xspect = data['spects']
xquant = data['quantized_audios']
xraw = data['quantized_audios']

print(xraw.shape, xspect.shape, xquant.shape)

x = torch.LongTensor(xquant)[:2]
x_batch = torch.nn.functional.one_hot(x, num_classes=256).float()
c_batch = torch.FloatTensor(xspect)[:2]

# chnage x to batch_size x channel x time
print(x_batch.shape)
x_batch = x_batch.permute((0, 2, 1))
print(x_batch.shape)
print(x_batch.dtype)

# batch local conditioning to batch_size x channel x time as well
print(c_batch.shape)
c_batch = c_batch.permute((0, 2, 1))
print(c_batch.shape)

model(x_batch, c_batch, None, False)