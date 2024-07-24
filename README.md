# IEEE Bigdata Cup 2024: Building extraction
IEEE BigData Cup Challenge 2024: Cross-City Generalizability of Instance Segmentation Model in a Nationwide Building Extraction Task
- Link: https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview

### Results
- Inference
  ```bash
  
  ``` 

### Experiments
Strategy 
- Identify a set of models for instance segmentation [MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets)
  - RTMDet-Instance
  - Approach [TTA](https://mmdetection.readthedocs.io/en/latest/user_guides/test.html?highlight=TTA#test-time-augmentation-tta) - https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_tta.py
- Finetune models - https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html
- Approach ensemble for improved detections
- Add other tricks - https://github.com/open-mmlab/mmyolo/blob/main/docs/en/recommended_topics/training_testing_tricks.md




### Environment
- Python 3.9 Installation via Miniconda - https://docs.conda.io/projects/miniconda/en/latest/ 
  ```bash
  conda env remove -n segm
  conda create -n segm python=3.9
  conda activate segm
  pip install -r requirements.txt
  ```
- MMDetection environment
  ```bash
  pip install -U openmim wandb future tensorboard prettytable
  mim install "mmengine>=0.6.0" "mmcv>=2.0.0rc4, <2.2.0"
  mim install albumentations --no-binary qudida,albumentations
  ```
- MMDetection installation
  ```
  git clone https://github.com/open-mmlab/mmdetection.git model/mmdetection
  cd model/mmdetection
  pip install -v -e .
  ```
- Kaggle CLI `pip install kaggle` - https://www.kaggle.com/docs/api#getting-started-installation-&-authentication
  - Obtain the kaggle.json by generating new token from accounts and place at ~/.kaggle/ path
  - Dataset download using `kaggle competitions download -c building-extraction-generalization-2024`
  

### Dataset
- Download data from [Kaggle ](https://www.kaggle.com/competitions/building-extraction-generalization-2024/data)
  ```bash
  tree -L 2 .
  .
  ├── sample_submission.csv
  ├── test
  │   └── image
  ├── train
  │   ├── image
  │   └── train.json
  └── val
      ├── image
      └── val.json
  ```
- Preparation
  - COCO format
  ```bash
  
  ```
