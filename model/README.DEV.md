### Exp

#### Run - RTMDet-M-Instance
```
# Multi 3 GPU # XXX hours for b16 100 epochs
CUDA_VISIBLE_DEVICES=4,6,7 PORT=29601 ./tools/dist_train.sh rtmdet_ins_m_beg.py 3
```
> 07/24 15:33:25 --> 
- training logs
  ```logs
  07/24 15:34:09 - mmengine - INFO - Epoch(train)   [1][50/79]  base_lr: 1.9623e-04 lr: 1.9623e-04  eta: 16:40:12  time: 7.6449  data_time: 0.0865  memory: 34938  loss: 3.0395  loss_cls: 1.7496  loss_bbox: 0.7522  loss_mask: 0.5377  
  ...
  
  ```