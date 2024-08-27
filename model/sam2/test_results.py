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
TEST_RESULT_PATH=os.path.join(PROJECT_BASE, "dataset/submissions/20240813-submissions-dets-masato66.csv")
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
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = torch.device("cuda")
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    mask_generator = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False))


    # get file list for iteration
    files = sorted(os.listdir(TEST_IMAGES_PATH))
    beg_results_dict = {"ImageID": [], "Coordinates": []}
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        print("Processing: {} files".format(len(files)))
        for file in tqdm(files):
            image_id = file[:-4]
            masks = None
            image = get_image(image_id)
            input_point, input_label = get_prompt_points(prompts_df, image_id, aggregate="center")  
            input_box, _ = get_prompt_points(prompts_df, image_id, aggregate = "rect")
            #print(input_box) # Input boxes are not detected in few cases
            
            if len(input_box)>0:
                # SamPredictor remembers this embedding
                predictor.set_image(image)
                input_boxes = torch.tensor(input_box, device=predictor.device)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                #masks = masks.squeeze(0)
                #print(file, "\t mask: ",masks.shape)
            else:
                # In case of no prompts SAM Automatic and should not hit in most cases. Eg: 0125.tif 
                masks = mask_generator.generate(image)
                print(file, "\t No input box available but still mask: ",masks.shape)

            pred_polygons = []
            if masks is not None and masks.ndim > 3:
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
            #predictor.reset_image()


    # Add these results to pandas dataframe
    results_df = pd.DataFrame({ 'ImageID': beg_results_dict["ImageID"], 'Coordinates': beg_results_dict["Coordinates"] })
    results_df = results_df.sort_values(by=["ImageID"], ascending=True)
    results_df.to_csv(os.path.join(TEST_OUTPUT_PATH, "..", "beg_test_tsam2.v1.csv"), index=False)    