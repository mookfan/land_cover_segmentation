import os

from numpy import random
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config.default import CfgNode
from dataset.patch_dataset import PatchDataset
from dataset.transforms import get_transform
from utils.utilities import build_dataset_stats_json_from_cfg, get_gpu_count


def get_dataloader(cfg: CfgNode, mode: str) -> DataLoader:

    if not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE):
        build_dataset_stats_json_from_cfg(cfg)

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(cfg, mode, transforms)

    num_workers = cfg.TRAIN.WORKERS
    shuffle = cfg.TRAIN.SHUFFLE

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * get_gpu_count(cfg),
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=random.seed(cfg.TRAIN.SEED),
        drop_last=True,
    )

    return dataloader
