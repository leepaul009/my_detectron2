
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
### 3.3 Change category in training
To Change the category size, edit the file detectron2\data\datasets\coco.py as follow:
```
def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
  ...
   VALID_CLASSES = ('Pedestrian','Cyclist','Car','Truck','Van')
```
and edit the file detectron2\data\datasets\builtin_meta.py:
```
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Pedestrian"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Cyclist"},
    {"color": [0, 0, 142],   "isthing": 1, "id": 3, "name": "Car"},
    {"color": [0, 0, 230],   "isthing": 1, "id": 4, "name": "Truck"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Van"},
]
```
or only consider Pedestrain:
```
def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
  ...
   VALID_CLASSES = ('Pedestrian')
```
and edit the file detectron2\data\datasets\builtin_meta.py:
```
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Pedestrian"},
]
```
In this case, data loader will consider all the images in validation set. Even only the category 'Pedestrain' is considered in training and some images do not contain category 'Pedestrain'.

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

