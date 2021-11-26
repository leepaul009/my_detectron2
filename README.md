
## 1. Prepare Datasets
annotations should be the format of COCO.
Note: Images in 'images/test' are validation set(sorry for bad naming).
```
./datasets/ped
├── annotations
│   └── dhd_traffic_train_good.json (remove two bad image which can not be read)
│   └── dhd_traffic_val.json
├── images
│   └── train
│   └── test
```

## 2. Installation
### 2.1 setup environment
```
conda create -n torch19-py38 python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pycocotools
pip install  tqdm scipy pandas
pip install opencv-python
cd detectron2
pip install -e . 
```

## 3 Train and inference
### 3.1 Download pre-trained model:
you can download model from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md.
or directlt download model from:
https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl

### 3.2 Train:
```
python tools/train_net.py    --num-gpus 4   --resume   --config-file configs/Ped/base.yaml   MODEL.WEIGHTS model_final_480dd8.pkl   OUTPUT_DIR "Experiments/cascade_mask_rcnn/r_50_norm"
```




## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

