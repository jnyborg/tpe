import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from distutils.util import strtobool
import json
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_evaluation_loaders
from evaluation import evaluation, validation
from models.stclassifier import PseGru, PseLTae, PseTae, PseTempCNN
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ShiftAug,
    ToTensor,
)
from utils.focal_loss import FocalLoss
from utils.metrics import overall_classification_report
from utils.train_utils import AverageMeter, bool_flag, to_cuda


def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior for conv
    device = torch.device(config.device)

    if config.overall:
        print('Source result')
        overall_performance(config, for_target=False)
        print('\nTarget result')
        overall_performance(config, for_target=True)
        return

    for fold_num in range(config.num_folds):
        print(f"Starting fold {fold_num}...")

        config.fold_dir = os.path.join(config.output_dir, f"fold_{fold_num}")
        config.fold_num = fold_num

        sample_pixels_val = config.sample_pixels_val or (
            config.eval and config.temporal_shift
        )
        print('creating source evaluation loaders')
        source_val_loader, source_test_loader = create_evaluation_loaders(
            config.source, config, sample_pixels_val
        )
        if config.eval_target:
            print('creating target evaluation loaders')
            target_val_loader, target_test_loader = create_evaluation_loaders(
                config.target, config, sample_pixels_val
            )
        else:
            target_val_loader, target_test_loader = None, None

        if config.model == "pseltae":
            model = PseLTae(
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                with_extra=config.with_extra,
                with_gdd_extra=config.with_gdd_extra,
                with_pos_enc=config.with_pos_enc,
                with_gdd_pos=config.with_gdd_pos,
                pos_type=config.pos_type,
            )
        elif config.model == "psetae":
            model = PseTae(
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                with_extra=config.with_extra,
                pos_type=config.pos_type
            )
        elif config.model == "psetcnn":
            model = PseTempCNN(
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                with_extra=config.with_extra,
            )
        elif config.model == "psegru":
            model = PseGru(
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                with_extra=config.with_extra,
            )
        else:
            raise NotImplementedError()

        model.to(config.device)

        best_model_path = os.path.join(config.fold_dir, "model.pt")

        if not config.eval:
            print(model)
            print("Number of trainable parameters:", get_num_trainable_params(model))

            if os.path.isfile(best_model_path):
                answer = input(
                    f"Model already exists at {best_model_path}! Override y/[n]? "
                )
                override = strtobool(answer) if len(answer) > 0 else False
                if not override:
                    print("Skipping fold", fold_num)
                    continue

            writer = SummaryWriter(
                log_dir=f"{config.tensorboard_log_dir}_fold{fold_num}", purge_step=0
            )
            train_supervised(
                model, config, writer, source_val_loader, target_val_loader, device, best_model_path
            )

        print("Restoring best model weights for testing...")

        state_dict = torch.load(best_model_path)["state_dict"]
        model.load_state_dict(state_dict)

        test_metrics = evaluation(
            model, source_test_loader, device, config.classes, mode="test"
        )
        save_results(test_metrics, config, for_target=False)

        print(
            f"Test result for {config.experiment_name} on source: accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}"
        )

        if config.eval_target:
            test_metrics = evaluation(
                model, target_test_loader, device, config.classes, mode="test"
            )

            print(
                f"Test result for {config.experiment_name} on target: accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}"
            )

            save_results(test_metrics, config, for_target=True)

    overall_performance(config, for_target=False)
    if config.eval_target:
        overall_performance(config, for_target=True)


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_supervised(
    model, config, writer, source_val_loader, target_val_loader, device, best_model_path
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            ShiftAug(max_shift=60, p=1.0) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    datasets = config.source
    if config.train_on_target:
        datasets = config.target

    train_dataset = PixelSetData(
        config.data_root,
        datasets,
        config.classes,
        train_transform,
        split='train',
        fold_num=config.fold_num,
    )

    if config.class_balance:
        source_labels = train_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using class balanced training loader")
        train_data_loader = data.DataLoader(
            train_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    elif config.tile_balance:
        source_tiles = train_dataset.get_tiles()
        freq = Counter(source_tiles)
        tile_weights = {x: 1.0 / freq[x] for x in freq}
        tile_weights = [tile_weights[x] for x in source_tiles]
        sampler = WeightedRandomSampler(tile_weights, len(source_tiles))
        print("using tile balanced training loader")
        train_data_loader = data.DataLoader(
            train_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        train_data_loader = data.DataLoader(
            dataset=train_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=config.batch_size,
            drop_last=True,
        )

    steps_per_epoch = min(config.steps_per_epoch, len(train_data_loader))
    print(f"training dataset: {datasets}, n={len(train_dataset)}, batches={len(train_data_loader)}, steps={steps_per_epoch}")

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0
    )

    best_f1_src, best_f1_trg = 0, 0

    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()

        train_iter = iter(train_data_loader)

        progress_bar = tqdm(
            range(steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
        )
        global_step = epoch * steps_per_epoch
        for step in progress_bar:
            sample = next(train_iter)
            targets = sample["label"].cuda(device=device, non_blocking=True)

            pixels, mask, positions, extra, gdd = to_cuda(sample, device)
            outputs = model.forward(pixels, mask, positions, extra, gdd)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(lr=f"{lr:.1E}", loss=f"{loss_meter.avg:.3f}")
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)

        progress_bar.close()

        model.eval()
        best_f1_src = validation(
            best_f1_src,
            best_model_path,
            config,
            criterion,
            device,
            epoch,
            model,
            source_val_loader,
            writer,
        )
        if config.eval_target:
            best_f1_trg = validation(
                best_f1_trg,
                None,
                config,
                criterion,
                device,
                epoch,
                model,
                target_val_loader,
                writer,
                prefix="val_target",
            )
        print()


def create_train_val_test_folds(
    datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2
):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train : n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
        folds.append(splits)
    return folds


def save_results(metrics, config, for_target=True):
    out_dir = config.fold_dir
    metrics = deepcopy(metrics)
    conf_mat = metrics.pop("confusion_matrix")
    class_report = metrics.pop("classification_report")
    if for_target:
        target_name = ','.join([str(tile).replace("/", "_") for tile in config.target])
    else:
        target_name = ','.join([str(tile).replace("/", "_") for tile in config.source])

    with open(
        os.path.join(out_dir, f"test_metrics_{target_name}.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    with open(os.path.join(out_dir, f"class_report_{target_name}.txt"), "w") as outfile:
        outfile.write(str(class_report))
    pkl.dump(conf_mat, open(os.path.join(out_dir, f"conf_mat_{target_name}.pkl"), "wb"))


def overall_performance(config, for_target=True):
    overall_metrics = defaultdict(list)
    if for_target:
        target_name = ','.join([str(tile).replace("/", "_") for tile in config.target])
    else:
        target_name = ','.join([str(tile).replace("/", "_") for tile in config.source])

    cms = []
    for fold in range(config.num_folds):
        fold_dir = os.path.join(config.output_dir, f"fold_{fold}")
        test_metrics = json.load(
            open(os.path.join(fold_dir, f"test_metrics_{target_name}.json"))
        )
        for metric, value in test_metrics.items():
            overall_metrics[metric].append(value)
        cm = pkl.load(open(os.path.join(fold_dir, f"conf_mat_{target_name}.pkl"), "rb"))
        cms.append(cm)

    for i, row in enumerate(np.mean(cms, axis=0)):
        print(config.classes[i], row.astype(int))

    print(f"Overall result across {config.num_folds} folds:")
    print(overall_classification_report(cms, config.classes))
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if metric == "loss":
            print(f"{metric}: {np.mean(values):.4}±{np.std(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.1f}±{np.std(values):.1f}")

    with open(
        os.path.join(config.output_dir, f"overall_{target_name}.json"), "w"
    ) as file:
        file.write(json.dumps(overall_metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument(
        "--data_root",
        default="/media/data/timematch_data",
        type=str,
        help="Path to datasets root directory",
    )
    parser.add_argument(
        "--num_blocks",
        default=100,
        type=int,
        help="Number of geographical blocks in dataset for splitting. Default 100.",
    )

    available_tiles = [
        "denmark/32VNH/2017",
        "france/30TXT/2017",
        "france/31TCJ/2017",
        "austria/33UVP/2017",
    ]

    parser.add_argument(
        "--source", default=["denmark/32VNH/2017"], nargs="+", choices=available_tiles
    )
    parser.add_argument(
        "--target", default=["france/30TXT/2017"], nargs="+", choices=available_tiles
    )
    parser.add_argument(
        "--num_folds", default=1, type=int, help="Number of train/test folds"
    )
    parser.add_argument(
        "--val_ratio",
        default=0.1,
        type=float,
        help="Ratio of training data to use for validation. Default 10%.",
    )
    parser.add_argument(
        "--test_ratio",
        default=0.2,
        type=float,
        help="Ratio of training data to use for testing. Default 20%.",
    )
    parser.add_argument(
        "--sample_pixels_val",
        type=bool_flag,
        default=True,
        help="speed up validation at the cost of randomness",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Path to the folder where the results should be stored",
    )
    parser.add_argument(
        "-e", "--experiment_name", default=None, help="Name of the experiment"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of data loading workers"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Name of device to use for tensor computations",
    )
    parser.add_argument(
        "--log_step",
        default=10,
        type=int,
        help="Interval in batches between display of training metrics",
    )
    parser.add_argument("--eval", action="store_true", help="run only evaluation")
    parser.add_argument(
        "--overall", action="store_true", help="print overall results, if exists"
    )
    parser.add_argument("--combine_spring_and_winter", default=False, type=bool_flag)

    # Training configuration
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs per fold"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--steps_per_epoch", default=500, type=int, help="Batches per epoch")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay rate"
    )
    parser.add_argument(
        "--focal_loss_gamma", default=1.0, type=float, help="gamma value for focal loss"
    )
    parser.add_argument(
        "--num_pixels",
        default=64,
        type=int,
        help="Number of pixels to sample from the input sample",
    )
    parser.add_argument(
        "--seq_length",
        default=30,
        type=int,
        help="Number of time steps to sample from the input sample",
    )
    parser.add_argument(
        "--model",
        default="pseltae",
        choices=["psetae", "pseltae", "psetcnn", "psegru"],
    )
    parser.add_argument(
        "--input_dim", default=10, type=int, help="Number of channels of input sample"
    )
    parser.add_argument(
        "--with_extra",
        default=False,
        type=bool_flag,
        help="whether to input extra geometric features to the PSE",
    )
    parser.add_argument(
        "--with_gdd_extra",
        default=False,
        type=bool_flag,
        help="whether to input extra gdd weather data",
    )
    parser.add_argument("--with_pos_enc", default=True, type=bool_flag)
    parser.add_argument("--with_gdd_pos", default=False, action="store_true")
    parser.add_argument("--with_shift_aug", default=False, action="store_true")
    parser.add_argument("--eval_target", default=True, type=bool_flag)
    parser.add_argument("--max_temporal_shift", default=60, type=int)
    parser.add_argument("--class_balance", default=False, action="store_true")
    parser.add_argument("--tile_balance", default=False, action="store_true")
    parser.add_argument("--tensorboard_log_dir", default="runs")
    parser.add_argument(
        "--train_on_target",
        default=False,
        action="store_true",
        help="supervised training on target for upper bound comparison",
    )
    parser.add_argument(
        "--pos_type",
        default="default",
        choices=['default', 'fourier', 'rnn'],
    )

    # Specific parameters for each training method
    config = parser.parse_args()

    # Setup classes

    # Dynamically load classes
    # dataset_classes = json.load(open(os.path.join(config.data_root, 'dataset_classes.json'), 'r'))
    # classes_in_datasets = [set(dataset_classes[d]) for d in config.source]
    # classes = sorted(list(set.intersection(*classes_in_datasets)))
    
    # Fixed set of classes with at least 200 examples in all tiles
    classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley', 'unknown', 'winter_barley', 'winter_rapeseed', 'winter_triticale', 'winter_wheat'])
    config.classes = classes
    config.num_classes = len(classes)
    print('Using classes', classes)

    # Setup folders based on name
    if config.experiment_name is not None:
        config.tensorboard_log_dir = os.path.join(
            config.tensorboard_log_dir, config.experiment_name
        )
        config.output_dir = os.path.join(config.output_dir, config.experiment_name)

    os.makedirs(config.output_dir, exist_ok=True)
    for fold in range(config.num_folds):
        os.makedirs(os.path.join(config.output_dir, "fold_{}".format(fold)), exist_ok=True)

    # write training config to file
    if not config.eval:
        with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
            f.write(json.dumps(vars(config), indent=4))
    print(config)
    main(config)
