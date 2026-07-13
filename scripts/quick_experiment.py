"""
Minimal train/validation experiment for AttnSleep.

Useful when you only have a subset of NPZ files or want a fast smoke test
without running the full 20-fold pipeline.
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn

from data_loader.data_loaders import data_generator_np
from model import loss as module_loss
from model import metric as module_metric
from model.model import AttnSleep
from trainer.trainer import Trainer
from parse_config import ConfigParser
from utils.util import calc_class_weight
from train_Kfold_CV import weights_init_normal


def build_subject_pairs(npz_dir):
    files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    grouped = {}
    for path in files:
        name = os.path.basename(path).replace(".npz", "")
        key = name[3:5]
        grouped.setdefault(key, []).append(path)
    return list(grouped.values())


def split_pairs(pairs, val_ratio=0.2, seed=123):
    rng = np.random.default_rng(seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_idx = set(order[:n_val])
    train_files, val_files = [], []
    for idx, pair in enumerate(pairs):
        if idx in val_idx:
            val_files.extend(pair)
        else:
            train_files.extend(pair)
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config_quick.json")
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--np_data_dir", default="data/edf_20_npz")
    parser.add_argument("--experiment_tag", default="baseline")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    with open(args.config, encoding="utf-8") as handle:
        config_dict = json.load(handle)

    config_dict["name"] = f"{config_dict['name']}_{args.experiment_tag}"
    config = ConfigParser(config_dict, fold_id=0)

    pairs = build_subject_pairs(args.np_data_dir)
    if len(pairs) < 2:
        raise RuntimeError(
            f"Need at least 2 subject pairs in {args.np_data_dir}; found {len(pairs)}."
        )

    train_files, val_files = split_pairs(pairs)
    batch_size = config["data_loader"]["args"]["batch_size"]
    train_loader, val_loader, counts = data_generator_np(train_files, val_files, batch_size)
    class_weights = calc_class_weight(counts)

    model = AttnSleep()
    model.apply(weights_init_normal)
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    optimizer = config.init_obj("optimizer", torch.optim, model.parameters())

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=train_loader,
        fold_id=0,
        valid_data_loader=val_loader,
        class_weights=class_weights,
    )
    trainer.train()


if __name__ == "__main__":
    main()
