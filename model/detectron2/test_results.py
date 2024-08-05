
import os, csv
from argparse import ArgumentParser
from pathlib import Path

import cv2 
import mmcv

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model



def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+", newline='', encoding='utf-8') as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)


# CUDA_VISIBLE_DEVICES=6 ../../tools/lazyconfig_train_net.py --config-file configs/mask_rcnn_mvitv2_t_3x_beg.py 
# --eval-only train.init_checkpoint=output_mask_rcnn_mvitv2_t_3x_beg_b8/model_final.pth
def main():
    args = default_argument_parser().parse_args()

    #--config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    #print(do_test(cfg, model))

    # get file list
    test_images_path = "/home/rahul/workspace/vision/beg24/building_extraction_generalization_2024/dataset/coco/test/image/"
    files = os.listdir(test_images_path)
    beg_results_dict = {"ImageID": [], "Coordinates": []}
    # evaluate model:
    model.eval()
    with torch.no_grad():
        test_dataloader = instantiate(cfg.dataloader.test)
        # start detector inference
        for idx, inputs in enumerate(test_dataloader):
            img_file = inputs[0]["file_name"]
            img = mmcv.imread(img_file)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            image_id = os.path.basename(img_file)
            print(idx, "\t", img_file)
            results = model(inputs)
            pred_instances = results[0]["instances"]
            pred_polygons  = []
            # Read each mask in prediction
            for pred_mask, score in zip(pred_instances.pred_masks, pred_instances.scores):
                masks = pred_mask.cpu().numpy().astype("uint8")
                if np.sum(masks==1) and score>0.5: # if mask is not empty - i.e. filled with zeros only, then proceed
                    masks_img = np.array(masks*255, dtype = np.uint8)
                    #print(masks_img.shape, np.unique(masks_img))
                    gray_img = Image.fromarray(masks_img, mode='L')
                    #print(gray_img.size)
                    contours, _ = cv2.findContours(image=np.array(gray_img, dtype = np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1 )
                    # Improve contour - Watershed or Arc approximation [https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/]
                    cv2.drawContours(img, contours, -1, (0,0,255), 1)

                    #contours = measure.find_contours(np.array(gray_img), level=0.8)
                    contour_per_pred = []
                    for item in contours: # Taking the first polygon bound 
                        #print("-->", len(item))
                        points = [tuple((pt[0][0], pt[0][1])) for pt in item]
                        contour_per_pred.extend(points)
                    # Add contour to the list
                    pred_polygons.append(contour_per_pred)
                    #print("\tScore:", score, "\tMasks(len):",  len(contour_per_pred), "\tMasks(5 pts):",  contour_per_pred[:5])

            # Result row item format as per https://www.kaggle.com/competitions/building-extraction-generalization-2024/data
            beg_results_dict["ImageID"].append(int(image_id[:-4]))
            beg_results_dict["Coordinates"].append(pred_polygons)
            mmcv.imwrite(img, os.path.join("./output", "pred_{}.jpg".format(image_id)))

        # Add these to pandas dataframe
        results_df = pd.DataFrame({ 'ImageID': beg_results_dict["ImageID"], 'Coordinates': beg_results_dict["Coordinates"] })
        results_df = results_df.sort_values(by=["ImageID"], ascending=True)
        results_df.to_csv(os.path.join("./output", "beg_test_d2.v2.csv"), index=False)
            


"""
10       /home/rahul/workspace/vision/beg24/building_extraction_generalization_2024/dataset/coco/test/image/0455.tif
[{'instances': Instances(num_instances=12, image_height=500, image_width=500, fields=[pred_boxes: Boxes(tensor([[1.8085e+02, 7.2031e-02, 4.9795e+02, 1.2330e+02],
        [2.8437e+02, 1.6928e+02, 3.2716e+02, 2.0347e+02],
        [2.1063e+02, 2.9816e+02, 2.5912e+02, 3.3956e+02],
        [2.4897e+02, 3.3420e+02, 2.7348e+02, 3.5544e+02],
        [2.5859e+02, 3.6343e+02, 2.7693e+02, 3.8070e+02],
        [4.2143e+02, 1.8402e+02, 5.0000e+02, 3.0440e+02],
        [1.1330e-01, 3.2992e+02, 2.9906e+01, 3.9283e+02],
        [2.3831e+02, 3.7005e+02, 2.6191e+02, 3.9296e+02],
        [4.6069e+02, 1.7356e+02, 5.0000e+02, 2.2198e+02],
        [4.3983e+02, 1.7374e+02, 4.9841e+02, 2.5466e+02],
        [4.7739e+02, 2.0262e+02, 5.0000e+02, 2.3520e+02],
        [2.4746e+02, 3.0178e+02, 2.7734e+02, 3.3261e+02]], device='cuda:0')), 
        
        scores: tensor([1.0000, 1.0000, 0.9999, 0.9999, 0.9990, 0.9985, 0.9973, 0.9900, 0.1645,
        0.1498, 0.1066, 0.1003], device='cuda:0'), 
        
        pred_classes: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0'), 
        
        pred_masks: tensor([[[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],
        ...
        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]]], device='cuda:0')])}]


"""
if __name__ == '__main__':
    main()
