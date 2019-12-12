import os, numpy as np, random, torch, torchvision, pickle
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import *
from pathlib import Path
from dataclasses import dataclass
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
n_classes = 256


def train_one_step(model, optimizer, example, global_step, device):
    global loss_fn, n_classes

    # y: BatchSize x Time, c: BatchSize x Channels x Time
    if len(example) > 1:
        y, c = example
        y = y.to(device)
        c = c.to(device)
    else:
        y = example
        y = y.to(device)
        c = None

    x = F.one_hot(y, num_classes=n_classes).float()
    # change to BatchSize x Channels x Time
    x = torch.transpose(x, 2, 1)

    y_pred = model(x, c, None, False)
    y_pred = y_pred.view(-1, n_classes)
    y = y.view(-1)

    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = torch.sum((torch.argmax(y_pred, dim=-1) == y).int()).float() / y.shape[0]
    return {"loss": loss, "accuracy": accuracy}


def train(train_dataset,
          model,
          optimizer,
          device,
          train_one_step,
          epoch_seeds: List[int],
          global_step: int,
          n_epoches: int,
          no_steps_per_epoch: int,
          ckpt_dir: str,
          train_history=None,
          eval_freq: int = float('inf'),
          eval_fn=None,
          save_freq: int = float('inf'),
          save_history_freq: int = float('inf'),
          log_freq: int = float('inf'),
          log_metrics: List[Any] = []):
    """
    @param eval_freq: number of epoches we will evaluate the model (can be fractional)
    @param eval_fn: optional, will be call when we run evaluation
    @param save_freq: number of epoches we will save the model (can be fractional)
    @param log_freq: number of iteration we will log information
    @param log_metrics:
    """
    Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
    model.train(True)

    for epoch in range((global_step // no_steps_per_epoch), n_epoches):
        for metric in log_metrics:
            metric.reset()

        set_seed(epoch_seeds[epoch + 1])

        pbar = tqdm(initial=epoch * no_steps_per_epoch,
                    total=n_epoches * no_steps_per_epoch,
                    desc='training')
        for train_example in train_dataset:
            global_step += 1
            pbar.update(1)

            loss = train_one_step(model, optimizer, train_example, global_step, device)
            # loss = model.loss(x, global_step)
            # optimizer.zero_grad()
            # loss['loss'].backward()
            # optimizer.step()

            # need to update metric
            for metric in log_metrics:
                if metric.name in loss:
                    metric.update(loss[metric.name].item())

            # logging
            if global_step % log_freq == 0:
                if train_history is not None:
                    train_history.push(loss, global_step)

                info = {metric.name: metric.value for metric in log_metrics}
                info['epoch'] = epoch
                pbar.set_postfix(**info)

        if (epoch + 1) % eval_freq == 0:
            model.train(False)
            eval_fn(model, global_step)
            model.train(True)

        if (epoch + 1) % save_freq == 0:
            _tmp = Path(ckpt_dir) / f"step_{global_step:09}"
            _tmp.mkdir(exist_ok=True)
            save_checkpoint(model, optimizer, global_step, epoch_seeds, str(_tmp / "model.bin"))

        if (epoch + 1) % save_history_freq == 0 and train_history is not None:
            train_history.flush(os.path.join(ckpt_dir, "history"), global_step)

    return global_step


def save_checkpoint(model, optim, global_step, epoch_seeds, file_path):
    states = {
        'iter': global_step,
        'model_states': model.state_dict(),
        'optim_states': optim.state_dict(),
        'seeds': epoch_seeds,
    }
    with open(file_path, mode='wb+') as f:
        torch.save(states, f)


def load_checkpoint(model, optim, file_path):
    if not os.path.isfile(file_path):
        raise Exception("no checkpoint found at '{}'".format(file_path))

    checkpoint = torch.load(file_path)
    global_iter = checkpoint['iter']
    seeds = checkpoint['seeds']
    model.load_state_dict(checkpoint['model_states'])

    if optim is not None:
        optim.load_state_dict(checkpoint['optim_states'])
    return global_iter, seeds


class TrainHistory:
    def __init__(self, names: List[str], scalar: List[str], scalars: List[str]):
        self.measurements = {name: [] for name in names}
        self.scalar = scalar
        self.scalars = scalars

    def push(self, loss, global_step):
        for name in self.scalar:
            if name in loss:
                self.add_scalar(name, loss[name].item(), global_step)
        for name in self.scalars:
            if name in loss:
                self.add_scalars(name, loss[name].cpu().detach().numpy(), global_step)

    def add_scalar(self, name: str, value, global_step):
        self.measurements[name].append([value, global_step])

    def add_scalars(self, name: str, values, global_step):
        self.measurements[name].append([values, global_step])

    def add_image(self, name: str, img, global_step):
        self.measurements[name].append([img, global_step])

    def flush(self, history_dir, global_step):
        Path(history_dir).mkdir(exist_ok=True)
        with open(os.path.join(history_dir, f"step_{global_step:09}.pkl"), "wb+") as f:
            pickle.dump(
                {
                    "measurements": self.measurements,
                    "scalar": self.scalar,
                    "scalars": self.scalars,
                    "step": global_step
                }, f)
        self.measurements = {name: [] for name in self.measurements.keys()}


class MeanMetric:
    def __init__(self, name: str):
        self.name = name
        self.total = 0.0
        self.count = 0

    def update(self, val):
        self.total += val
        self.count += 1

    @property
    def value(self):
        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0