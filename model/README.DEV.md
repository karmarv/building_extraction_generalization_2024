## Inference & Submissions
Exported the submissions CSV in the following order
> cd ./mmdetection
- [0.55853] - MM v1 `python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_m_beg.py  ./work_dirs/rtmdet_ins_m_beg/epoch_100.pth  --out-dir ./work_dirs/rtmdet_ins_m_beg/beg_test/  --to-labelme `
- [0.57086] - MM v2 `python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_s_beg.py  ./work_dirs/rtmdet_ins_s_beg/epoch_300.pth  --out-dir ./work_dirs/rtmdet_ins_s_beg/beg_test/  --to-labelme `
- [0.56499] - MM v3 `python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_m_beg.py  ./work_dirs/rtmdet_ins_m_beg/epoch_300.pth  --out-dir ./work_dirs/rtmdet_ins_m_beg/beg_test/  --to-labelme `

> cd ./detectron2/projects/MViTv2
- [Error] - D2 V1 `CUDA_VISIBLE_DEVICES=6 python test_results.py --config-file projects/MViTv2/configs/mask_rcnn_mvitv2_t_3x_beg.py --eval-only train.init_checkpoint=projects/MViTv2/output_mask_rcnn_mvitv2_t_3x_beg_b8/model_final.pth`
- [Error] - D2 V2 `CUDA_VISIBLE_DEVICES=6 python test_results.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_b_3x_beg.py --eval-only train.init_checkpoint=projects/MViTv2/output_cascade_mask_rcnn_mvitv2_b_3x_beg_b4/model_final.pth`

> SAM - Refine bounding box using prompts from submission CSV file
- [] - MMV2 SAM `CUDA_VISIBLE_DEVICES=6 time python test_results.py`
- [] - D2V2 SAM `CUDA_VISIBLE_DEVICES=6 time python test_results.py` 

> Refining team submissions
- [] - Pierre SAM "20240810-submission_a-pierre65.csv"            > Submit this
- [] - Masato SAM "20240809-submissions-dets-masato66.csv"


Detectron2
CUDA_VISIBLE_DEVICES=6 ../../tools/lazyconfig_train_net.py --config-file configs/mask_rcnn_mvitv2_t_3x_beg.py --eval-only train.init_checkpoint=output_mask_rcnn_mvitv2_t_3x_beg_b8/model_final.pth


## (A.) MMDetection [./mmdetection](./mmdetection)

#### Run - Mask2Former 
```
# Multi 4 GPU # XXX hours for b20 300 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29601 ./tools/dist_train.sh mask2former_swin_be.py 4
```

#### Run - RTMDet-S-Instance
```
# Multi 4 GPU # XXX hours for b12 300 epochs
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_s_beg.py 4
```
>
- training validation peaked at epoch 300 - https://wandb.ai/karmar/beg24-mm/runs/qethaunx
```log
Evaluate annotation type *bbox*
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| building | 0.554 | 0.808  | 0.647  | 0.453 | 0.676 | 0.692 |
+----------+-------+--------+--------+-------+-------+-------+
Evaluate annotation type *segm*
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| building | 0.522 | 0.804  | 0.607  | 0.389 | 0.655 | 0.707 |
+----------+-------+--------+--------+-------+-------+-------+

08/01 03:41:24 - mmengine - INFO - Epoch(val) [300][20/20]    coco/building_precision: 0.5220  coco/bbox_mAP: 0.5540  coco/bbox_mAP_50: 0.8080  coco/bbox_mAP_75: 0.6470  coco/bbox_mAP_s: 0.4530  coco/bbox_mAP_m: 0.6760  coco/bbox_mAP_l: 0.6920  coco/segm_mAP: 0.5220  coco/segm_mAP_50: 0.8040  coco/segm_mAP_75: 0.6070  coco/segm_mAP_s: 0.3890  coco/segm_mAP_m: 0.6550  coco/segm_mAP_l: 0.7070  data_time: 0.0166  time: 0.6272
```

#### Run - RTMDet-M-Instance
```
# Multi 3 GPU # XXX hours for b12 100 epochs
CUDA_VISIBLE_DEVICES=4,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_m_beg.py 3
```
> Training Time: 07/24 15:33:25 --> 07/25 05:03:35
- training validation peaked at epoch 90 - https://wandb.ai/karmar/beg24-mm/runs/y4g1olrn
```log
Evaluate annotation type *bbox*
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| building | 0.548 | 0.8    | 0.636  | 0.449 | 0.669 | 0.653 |
+----------+-------+--------+--------+-------+-------+-------+
Evaluate annotation type *segm*
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| building | 0.517 | 0.793  | 0.607  | 0.384 | 0.656 | 0.683 |
+----------+-------+--------+--------+-------+-------+-------+
07/25 04:30:58 - mmengine - INFO - Epoch(val) [90][26/26]    coco/building_precision: 0.5170  coco/bbox_mAP: 0.5480  coco/bbox_mAP_50: 0.8000  coco/bbox_mAP_75: 0.6360  coco/bbox_mAP_s: 0.4490  coco/bbox_mAP_m: 0.6690  coco/bbox_mAP_l: 0.6530  coco/segm_mAP: 0.5170  coco/segm_mAP_50: 0.7930  coco/segm_mAP_75: 0.6070  coco/segm_mAP_s: 0.3840  coco/segm_mAP_m: 0.6560  coco/segm_mAP_l: 0.6830  data_time: 0.0152  time: 0.7167
...
07/25 05:03:35 - mmengine - INFO - Epoch(val) [100][26/26]    coco/building_precision: 0.5180  coco/bbox_mAP: 0.5410  coco/bbox_mAP_50: 0.7990  coco/bbox_mAP_75: 0.6330  coco/bbox_mAP_s: 0.4460  coco/bbox_mAP_m: 0.6580  coco/bbox_mAP_l: 0.6450  coco/segm_mAP: 0.5180  coco/segm_mAP_50: 0.7930  coco/segm_mAP_75: 0.6060  coco/segm_mAP_s: 0.3860  coco/segm_mAP_m: 0.6530  coco/segm_mAP_l: 0.6770  data_time: 0.0153  time: 0.7100
```

#### Run - RTMDet-L-Instance
```
# Multi 2 GPU # XXX hours for b12 200 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_l_beg.py 2
```

#### Run - RTMDet-X-Instance






## (B.) Detectron2 [./detectron2](./detectron2)
All configs can be trained with:
- `pip install timm==0.4.12` # Do not use fix_timm_model_layers file from MAE
```bash
# Install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

#### Run - ViTDet 
```
cd ./detectron2/projects/ViTDet
CUDA_VISIBLE_DEVICES=4 ../../tools/lazyconfig_train_net.py --config configs/COCO/mask_rcnn_vitdet_b_100ep_beg.py
```
  - epoch 100 training
  ```
  [08/05 00:51:44 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 55.111 | 80.552 | 63.306 | 45.362 | 66.867 | 70.534 |
  [08/05 00:51:47 d2.evaluation.coco_evaluation]: Evaluation results for segm:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 55.380 | 80.492 | 63.252 | 42.700 | 67.634 | 71.464 |
  ```

#### Run - MViT2
- Tiny
```
cd ./detectron2/projects/MViTv2
CUDA_VISIBLE_DEVICES=5 ../../tools/lazyconfig_train_net.py --config configs/mask_rcnn_mvitv2_t_3x_beg.py
```
  - Training epoch 100
  ```

  ```
  - Inference
  ```
  CUDA_VISIBLE_DEVICES=6 python test_results.py --config-file projects/MViTv2/configs/mask_rcnn_mvitv2_t_3x_beg.py --eval-only train.init_checkpoint=projects/MViTv2/output_mask_rcnn_mvitv2_t_3x_beg_b8/model_final.pth
  ```
- Base
```
cd ./detectron2/projects/MViTv2
CUDA_VISIBLE_DEVICES=5 ../../tools/lazyconfig_train_net.py --config configs/cascade_mask_rcnn_mvitv2_b_3x_beg.py
```
  - Training epoch 
  ```
  [08/04 09:28:55 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 57.918 | 79.891 | 66.135 | 48.049 | 69.837 | 71.994 |
  [08/04 09:28:58 d2.evaluation.coco_evaluation]: Evaluation results for segm:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 55.249 | 79.828 | 63.713 | 42.622 | 67.512 | 70.429 |
  ```
  - Inference
  ```
  CUDA_VISIBLE_DEVICES=6 python test_results.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_b_3x_beg.py --eval-only train.init_checkpoint=projects/MViTv2/output_cascade_mask_rcnn_mvitv2_b_3x_beg_b4/model_final.pth
  ```



## (C.) SAM [./segment-anything](./segment-anything)

- Install Python and CUDA 11.8
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# Ensure you dont install default and provide the /usr/local/cuda-11.8/ path in the install options
sudo sh cuda_11.8.0_520.61.05_linux.run
```
  - Environment variables
  ```
  #export CUDA_HOME="/usr/local/cuda"     # 12.4
  export CUDA_HOME="/usr/local/cuda-12.1"
  #export CUDA_HOME="/usr/local/cuda-11.8"
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
  - Test
  ```
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
  ```
  - Python environment
  ```
  conda env remove -n sam
  conda create -n sam python=3.10
  # Ensure CUDA 11.8
  pip install -r ../../requirements.txt
  ```

```
CUDA_VISIBLE_DEVICES=5 
python scripts/amg.py --checkpoint sam_vit_b_01ec64.pth --model-type vit_b --input /home/rahul/workspace/vision/beg24/building_extraction_generalization_2024/dataset/coco/test/image/ --output beg_test_sam/
```


### SAM2 [./sam2](./sam2)

  - Environment variables
  ```
  export CUDA_HOME="/usr/local/cuda-12.1"     # 12.1
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

  - Python environment
  ```
  conda env remove -n sam2
  conda create -n sam2 python=3.10 -y
  # Ensure CUDA 12.1
  pip install -r requirements.txt
  # verify
  python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME)'
  ```
  - Install Sam2
  ```
  pip install -e .
  pip install -e ".[demo]"
  ```