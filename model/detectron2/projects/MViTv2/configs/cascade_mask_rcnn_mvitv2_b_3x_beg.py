from .cascade_mask_rcnn_mvitv2_t_3x import model, dataloader, optimizer, lr_multiplier, train


from .common.coco_loader import dataloader

"""
1. Setup Dataset in COCO format
"""
import os
data_root = '/home/rahul/workspace/vision/beg24/building_extraction_generalization_2024/dataset/coco/'

from detectron2.data.datasets import register_coco_instances
register_coco_instances("beg_train", {}, os.path.join(data_root,"train/train.json"), 
                                            os.path.join(data_root,"train"))
register_coco_instances("beg_val", {}, os.path.join(data_root,"val/val.json"), 
                                            os.path.join(data_root,"val"))
register_coco_instances("beg_test", {}, os.path.join(data_root,"test/test.json"), 
                                            os.path.join(data_root,"test"))


model.backbone.bottom_up.depth = 24
model.backbone.bottom_up.last_block_indexes = (1, 4, 20, 23)
model.backbone.bottom_up.drop_path_rate = 0.4

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in1k.pyth"

# Data loaders
dataloader.train.dataset.names = "beg_train"
dataloader.test.dataset.names  = "beg_test"      # "beg_val"  # Changed for test report submission
dataloader.train.total_batch_size = 4