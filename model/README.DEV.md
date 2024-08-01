### Experiments

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



### Inference & Submissions

- [0.55853] - v1 `python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_m_beg.py  ./work_dirs/rtmdet_ins_m_beg/epoch_100.pth  --out-dir ./work_dirs/rtmdet_ins_m_beg/beg_test/  --to-labelme `
- [] - v2 `python test_results.py  ../../dataset/coco/test/image  ./rtmdet_ins_s_beg.py  ./work_dirs/rtmdet_ins_s_beg/epoch_300.pth  --out-dir ./work_dirs/rtmdet_ins_s_beg/beg_test/  --to-labelme `