import argparse
from typing import List
import os

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from models.models_utils import (
    rename_ordered_dict_from_parallel,
    rename_ordered_dict_to_parallel,
)
from train_utils import load_checkpoint
from utils.utilities import get_gpu_count
from utils.infer_utils import generate_outputs, prepare_raster_for_inference
from utils.io_utils import get_lines_from_txt

cfg_path = 'config/weighted_loss_more_snow_data_aug_hrnet.yml'
checkpoint = 'weights/cfg_weighted_loss_more_snow_data_aug_hrnet_3bands_resume_best_f1.pth'
samples_list_path = 'rasters/S2A_2022-01-01_2022-01-31_median.txt'
destination = 'debug_results/S2A_2022-01-01_2022-01-31_median/'
output_types = ['alphablend','raster','alphablended_raster', 'raw_raster']
aerial=False

cfg = get_cfg_from_file(cfg_path)
device = cfg.TEST.DEVICE

def infer(
    model: Module,
    dataloader: DataLoader,
    output_types: List[str],
    destination: str,
):
    """Evaluates test dataset and saves predictions if needed

    Args:
        model (Module): Model to use for inference
        dataloader (DataLoader): Dataloader for inference
        output_types (List[str]): List of output types.
                                  Supported types:
                                    * alphablend (img and predicted mask)
        destination (str): Path to save results

    Returns:
        dict: Generates and saves predictions in desired format
    """
    with torch.no_grad():
        model.eval()
        mask_config = dataloader.dataset.mask_config
        for batch in dataloader:
            inputs, names = batch["input"], batch["name"]

            # Forward propagation
            outputs = model(inputs)["out"]

            masks = torch.argmax(outputs, dim=1)

            for input_img, mask, name in zip(inputs, masks, names):

                generate_outputs(
                    output_types,
                    destination,
                    input_img,
                    mask,
                    name,
                    mask_config,
                    dataloader,
                )

if cfg.TEST.WORKERS > 0:
    torch.multiprocessing.set_start_method("spawn", force=True)

_, weights, _, _, _ = load_checkpoint(checkpoint, device)

model = get_model(cfg, device)
if get_gpu_count(cfg, mode="train") > 1 and get_gpu_count(cfg, mode="test") == 1:
    weights = rename_ordered_dict_from_parallel(weights)
if get_gpu_count(cfg, mode="train") == 1 and get_gpu_count(cfg, mode="test") > 1:
    weights = rename_ordered_dict_to_parallel(weights)
model.load_state_dict(weights)

if samples_list_path not in ["train", "val", "test"]:
    samples_list = get_lines_from_txt(samples_list_path)
    samples_to_infer = []
    # crop raster into sub-raster
    for sample_path in samples_list:
        cropped_samples_paths = prepare_raster_for_inference(
            sample_path, crop_size=[256, 256]
        )
        samples_to_infer.extend(cropped_samples_paths)

    with open(cfg.TEST.INFER_SAMPLES_LIST_PATH, "w") as f:
        for file in samples_to_infer:
            f.write(file + "\n")

    samples_list_path = cfg.TEST.INFER_SAMPLES_LIST_PATH

dataloader = get_dataloader(cfg, samples_list_path, aerial=aerial) # preprocess & normalize sentinel2 images
if not os.path.isdir(destination):
    os.makedirs(destination)
infer(
        model,
        dataloader,
        output_types,
        destination,
    )
print()