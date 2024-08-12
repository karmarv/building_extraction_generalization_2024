import os
import ast
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys
sys.path.append(".")

PROJECT_BASE="../../"
TEST_IMAGES_PATH=os.path.join(PROJECT_BASE, "dataset/coco/test/image/") # 0000.tif
TEST_RESULT_PATH=os.path.join(PROJECT_BASE, "dataset/submissions/20240810-submission_a-pierre65.csv")
TEST_OUTPUT_PATH = "./output/images"

os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)

def get_image(image_id):
    #image_id = "0000"
    image = cv2.imread(os.path.join(TEST_IMAGES_PATH, "{}.tif".format(image_id)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_prompt_points(prompts_df, image_id, aggregate):
    index = int(image_id)
    input_list  = prompts_df[prompts_df["ImageID"]==index]["Coordinates"]
    points, labels = [], []
    for item in input_list[index]:
        if aggregate == "center":
            in_point = np.mean(np.array(item), axis=0, dtype=np.int16)
            points.append([in_point[0], in_point[1]])
            labels.append(1)
        elif aggregate == "rect":
            ctr = np.array(item).reshape((-1,1,2)).astype(np.int32)
            x,y,w,h = cv2.boundingRect(ctr)
            points.append([x, y, x+w, y+h])
            labels.append(1)
        else:
            in_point = np.array(item)
            points.append(in_point)
            labels.append(np.ones(len(in_point)))
    return np.array(points), np.array(labels)

"""

CUDA_VISIBLE_DEVICES=6 time python test_results.py

"""
if __name__ == "__main__":

    prompts_df = pd.read_csv(TEST_RESULT_PATH, quotechar='"', sep=',')
    prompts_df['Coordinates'] = prompts_df['Coordinates'].apply(lambda x: ast.literal_eval(x))
    prompts_df.head()
        
    # !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam)
    """
    # For more fine segments
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    """

    # get file list for iteration
    files = sorted(os.listdir(TEST_IMAGES_PATH))
    beg_results_dict = {"ImageID": [], "Coordinates": []}
    
    for file in tqdm(files):
        image_id = file[:-4]

        image = get_image(image_id)
        input_point, input_label = get_prompt_points(prompts_df, image_id, aggregate="center")  
        input_box, _ = get_prompt_points(prompts_df, image_id, aggregate = "rect")
        #print(input_box) # Input boxes are not detected in few cases
        
        if len(input_box)>0:
            # SamPredictor remembers this embedding
            predictor.set_image(image)
            """
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            ) 
            # Choose the best mask based on score
            mask_logits = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            """
            input_boxes = torch.tensor(input_box, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.cpu().numpy()
            print(file, "\t mask: ",masks.shape)
        else:
            # In case of no prompts SAM Automatic and should not hit in most cases. Eg: 0125.tif 
            auto_masks = mask_generator.generate(image)
            #print("Count:",len(auto_masks), "\t Keys:",  auto_masks[0].keys())
            masks = []
            for m in auto_masks:
                masks.append(m["segmentation"].reshape(1, m["segmentation"].shape[0], m["segmentation"].shape[1]))
            print(file, "\t mask: ",len(masks), masks[0].shape, np.unique(masks[0]))

        pred_polygons = []
        for idx, mask_item in enumerate(masks):
            if np.sum(mask_item==1):
                mask_img = np.array(mask_item*255, dtype = np.uint8)
                #print(mask_img.shape, np.unique(mask_img)) # Expected (1, 500, 500) [  0 255]
                gray_img = Image.fromarray(mask_img[0], mode='L')
                #print(gray_img.size, np.unique(gray_img))
                contours, _ = cv2.findContours(image=np.array(gray_img, dtype = np.uint8),
                                                mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_TC89_L1 )                
                # Get max contour area
                cont = max(contours, key = cv2.contourArea)
                points = [tuple((pt[0][0], pt[0][1])) for pt in cont]
                # converting list to array
                points_arr = np.array(points)
                cv2.polylines(image,[points_arr.reshape((-1,1,2))], True, (0,255,0), 1)
                cv2.drawContours(image, cont, -1, (0,0,255), 1)  
                # Add contour to the list
                pred_polygons.append(points)  

        # Result row item format as per https://www.kaggle.com/competitions/building-extraction-generalization-2024/data
        beg_results_dict["ImageID"].append(int(image_id))
        beg_results_dict["Coordinates"].append(pred_polygons)
        cv2.imwrite(os.path.join(TEST_OUTPUT_PATH, "pred_{}.jpg".format(image_id)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # reset_image
        predictor.reset_image()


    # Add these results to pandas dataframe
    results_df = pd.DataFrame({ 'ImageID': beg_results_dict["ImageID"], 'Coordinates': beg_results_dict["Coordinates"] })
    results_df = results_df.sort_values(by=["ImageID"], ascending=True)
    results_df.to_csv(os.path.join(TEST_OUTPUT_PATH, "..", "beg_test_psam.v1.csv"), index=False)    