import json
import pathlib

from PIL import Image

import os

coco = {
    "images": [ ],
    "annotations": [ ],
    "categories":
    [
        {
            "id": 0,
            "name": "building",
            "supercategory": "building"
        }
    ]
}

def add_images_to_coco(image_dir, coco_filename):
    image_filenames = list(pathlib.Path(image_dir).glob('*.tif'))
    images = []
    for idx, image_filename in enumerate(image_filenames):
        im = Image.open(image_filename)
        width, height = im.size
        image_details = {
            "id": idx + 1,
            "height": height,
            "width": width,
            "file_name": "image/"+str(image_filename.name),
        }
        images.append(image_details)

    # This will overwrite the image tags in the COCO JSON file
    #with open(coco_filename) as f:
    #    data = json.load(f)

    coco['images'] = images

    with open(coco_filename, 'w') as coco_file:
        json.dump(coco, coco_file, indent = 4)

data_root = '/home/rahul/workspace/vision/beg24/building_extraction_generalization_2024/dataset/coco/'
add_images_to_coco(os.path.join(data_root, "test", "image"), os.path.join(data_root, "test", "test.json"))