
import os, csv
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import cv2 

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmdet.evaluation import get_classes


"""
python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_m_beg.py  ./work_dirs/rtmdet_ins_m_beg/epoch_100.pth  --out-dir ./work_dirs/rtmdet_ins_m_beg/beg_test/  --to-labelme 

"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    args = parser.parse_args()
    return args


def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+", newline='', encoding='utf-8') as my_csv:
        csvw = csv.writer(my_csv, delimiter=delimiter)
        csvw.writerows(rows)

def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    # get file list
    files = os.listdir(args.img)
    # get model class name
    dataset_classes = model.dataset_meta.get('classes')
    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    beg_results_dict = {"ImageID": [], "Coordinates": []}
    
    # start detector inference
    for file in tqdm(files):
        img_file = os.path.join(args.img, file)
        result = inference_detector(model, img_file)

        img = mmcv.imread(img_file)
        #img = mmcv.imconvert(img, 'bgr', 'rgb')

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        if args.to_labelme:
            #print(pred_instances)
            # Add results to CSV file
            polygon = []
            image_id = os.path.basename(file)
            for idx, pred in enumerate(pred_instances):
                masks = pred.masks.cpu().numpy().astype("uint8")
                if np.sum(masks==1): # if mask is not empty - i.e. filled with zeros only, then proceed
                    masks_img = np.array(masks*255, dtype = np.uint8)
                    gray_img = Image.fromarray(masks_img[0], mode='L')
                    #print(gray_img.size, np.unique(masks_img))
                    contours, _ = cv2.findContours(image=np.array(gray_img, dtype = np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1 )
                    # Improve contour - Watershed or Arc approximation [https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/]
                    cv2.drawContours(img, contours, -1, (0,0,255), 1)

                    #contours = measure.find_contours(np.array(gray_img), level=0.8)
                    contour_per_pred = []
                    for idx, item in enumerate(contours): # Taking the first polygon bound 
                        #print(idx, "-->", len(item))
                        points = [tuple((pt[0][0], pt[0][1])) for pt in item]
                        contour_per_pred.extend(points)
                    # Add contour to the list
                    polygon.append(contour_per_pred)
                    scores = pred.scores.tolist()
                    #bboxes = pred.bboxes.tolist()
                    #labels = pred.labels.tolist()
                    #print(" Score:", scores[0], "\tMasks(len):",  len(contour_per_pred), "\tMasks(5 pts):",  contour_per_pred[:5])
                    
            # Result row item format as per https://www.kaggle.com/competitions/building-extraction-generalization-2024/data
            #beg_results.append([image_id[:-4], polygon])
            beg_results_dict["ImageID"].append(int(image_id[:-4]))
            beg_results_dict["Coordinates"].append(polygon)
            
            
            if len(polygon)>0:
                mmcv.imwrite(img, os.path.join(args.out_dir, "pred_{}.jpg".format(image_id)))
                #print(beg_results_dict)
                #exit(0)
            continue

    if args.to_labelme:
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')
        #write_list_file(os.path.join(args.out_dir, "..", "beg_test.csv"), beg_results)
        # TODO: Add these to pandas dataframe
        results_df = pd.DataFrame({ 'ImageID': beg_results_dict["ImageID"], 'Coordinates': beg_results_dict["Coordinates"] })
        results_df = results_df.sort_values(by=["ImageID"], ascending=True)
        results_df.to_csv(os.path.join(args.out_dir, "..", "beg_test_l1.df.csv"), index=False)


if __name__ == '__main__':
    main()
