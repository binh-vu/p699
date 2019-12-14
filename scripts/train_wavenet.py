import argparse
import glob
import os
from pathlib import Path

import ujson

import torch, random
from torch import optim

from config import ROOT_DIR
from wavenet_vocoder.util import is_mulaw_quantize, is_scalar_input
from wavenet_vocoder import WaveNet
from wavenet_vocoder.data_loader import get_data_loader
from wavenet_vocoder.train import train, train_one_step, MeanMetric, TrainHistory, load_checkpoint
from wavenet_vocoder.tfcompat.hparam import HParams

hparams = HParams(
    name="wavenet_vocoder",
    input_type="mulaw-quantize",
    quantize_channels=256,  # 65536 or 256
    sample_rate=16000,
    num_mels=80,
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
    layers=18 * 2,
    stacks=2*2,
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
        "upsample_scales": [4, 4, 4, 2],  # should np.prod(upsample_scales) == hop_size (128)
    },

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker dataset
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=7,  # 7 for CMU ARCTIC

    # initial learning rate
    lr=1e-3,
    n_epoches=10,
    n_split=8,
    max_time_steps=None,
)


def get_model():
    global hparams
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
    # print(model)
    return model


def main(metadata, data_dir, ckpt_dir):
    quant_files = sorted(glob.glob(os.path.join(data_dir, "quantized_audios", "*.npz")))
    spect_files = sorted(glob.glob(os.path.join(data_dir, "spects", "*.npz")))
    assert len(quant_files) > 0

    train_dataset = get_data_loader(quant_files, spect_files, batch_size=1, shuffle=True, n_split=hparams.n_split,
                                    local_conditioning=True, max_time_steps=hparams.max_time_steps)

    device = torch.device('cuda')
    model = get_model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, amsgrad=True)

    random.seed(1212)
    seeds = [random.randint(0, 1212) for i in range(10000)]

    assert os.path.exists(ckpt_dir)
    ckpts = [x for x in Path(ckpt_dir).iterdir() if x.name.startswith("step_")]
    if len(ckpts) > 0:
        last_ckpt = max(ckpts, key=lambda x: int(x.name.replace("step_", "")))
        print(">>> restore last checkpoint", last_ckpt.name)
        global_step = load_checkpoint(model, optimizer, str(last_ckpt / "model.bin"))[0]
    else:
        global_step = 0

    global_step = train(train_dataset, model, optimizer,
                        device, train_one_step, init_lr=hparams.lr,
                        epoch_seeds=seeds, global_step=global_step, n_epoches=hparams.n_epoches,
                        no_steps_per_epoch=len(train_dataset),
                        ckpt_dir=str(ckpt_dir),
                        log_freq=2, log_metrics=[MeanMetric(x) for x in ['loss', 'accuracy', 'lr']],
                        train_history=TrainHistory(["loss", "accuracy", 'lr'], []),
                        save_freq=1, save_history_freq=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",
                        "-m",
                        required=True,
                        help="metadata file that contains information about training files and testing files")
    parser.add_argument("--data_dir",
                        "-i",
                        required=True,
                        help="dataset directory")
    parser.add_argument("--ckpt_dir",
                        "-c",
                        required=True,
                        help="dataset directory")
    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        metadata = ujson.load(f)

    main(metadata, args.data_dir, args.ckpt_dir)
