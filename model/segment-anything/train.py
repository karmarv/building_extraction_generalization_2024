# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

import sam_dataset as dataset
from sam_model import DiceLoss, FocalLoss
from sam_model import Model
from torch.utils.data import DataLoader
from sam_model import AverageMeter, calc_iou

from sam_model import show_box, show_mask

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()




cfg = {
    "num_devices": 4,
    "batch_size": 12,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "./out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/rahul/workspace/vision/building_extraction_generalization_2024/dataset/coco/train/",
            "annotation_file": "/home/rahul/workspace/vision/building_extraction_generalization_2024/dataset/coco/train/train.json"
        },
        "val": {
            "root_dir": "/home/rahul/workspace/vision/building_extraction_generalization_2024/dataset/coco/val/",
            "annotation_file": "/home/rahul/workspace/vision/building_extraction_generalization_2024/dataset/coco/val/val.json"            
        }
    }
}


def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = Model(cfg)
    model.setup()

    train_data, val_data = dataset.load_datasets(cfg, model.model.image_encoder.img_size)

    for step, ( images, bboxes, masks, images_path) in enumerate(train_data):
        idx = random.randint(0, 7)
        print("Image: ", images[idx])    
        print("Label: ", bboxes[idx])
        print("Paths: ", images_path[idx])
        
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        
        axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[0])
        show_box(bboxes[idx].numpy(), axs[0])
        axs[0].axis("off")
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(0, 7)
        axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[1])
        show_box(bboxes[idx].numpy(), axs[1])
        axs[1].axis("off")
        # set title
        axs[1].set_title(names_temp[idx])
        # plt.show()
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
        plt.close()
        break

    return

if __name__ == "__main__":
    main(cfg)