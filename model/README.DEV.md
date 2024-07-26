### Exp

#### Run - RTMDet-M-Instance
```
# Multi 3 GPU # XXX hours for b12 100 epochs
CUDA_VISIBLE_DEVICES=4,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_m_beg.py 3
```
> Training Time: 07/24 15:33:25 --> 07/25 05:03:35
- training validation peaked at epoch 80 - https://wandb.ai/karmar/beg24-mm/runs/y4g1olrn
  ```logs
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.548
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.800
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.636
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.669
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.021
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.172
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.504
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.764
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.814
  07/25 04:30:14 - mmengine - INFO -
  +----------+-------+--------+--------+-------+-------+-------+
  | category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
  +----------+-------+--------+--------+-------+-------+-------+
  | building | 0.548 | 0.8    | 0.636  | 0.449 | 0.669 | 0.653 |
  +----------+-------+--------+--------+-------+-------+-------+

  ...
  07/25 05:03:35 - mmengine - INFO - Epoch(val) [100][26/26]    coco/building_precision: 0.5180  coco/bbox_mAP: 0.5410  coco/bbox_mAP_50: 0.7990  coco/bbox_mAP_75: 0.6330  coco/bbox_mAP_s: 0.4460  coco/bbox_mAP_m: 0.6580  coco/bbox_mAP_l: 0.6450  coco/segm_mAP: 0.5180  coco/segm_mAP_50: 0.7930  coco/segm_mAP_75: 0.6060  coco/segm_mAP_s: 0.3860  coco/segm_mAP_m: 0.6530  coco/segm_mAP_l: 0.6770  data_time: 0.0153  time: 0.7100
  ```

#### Run - RTMDet-L-Instance
```
# Multi 2 GPU # XXX hours for b12 200 epochs
CUDA_VISIBLE_DEVICES=6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_l_beg.py 2
```

#### Run - RTMDet-X-Instance
