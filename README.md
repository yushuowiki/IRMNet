# IRMNet
This is the implementation of our work `Few-shot Semantic Segmentation Based on Inter-class Relation Mining`. 

# Get Started

### Environment
+ torch==1.8.0
+ numpy==1.21.5
+ tensorboardX==2.5
+ cv2==4.5.5.64


### Datasets and Data Preparation

Please download the following datasets:

+ PASCAL-5i is based on the [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html) where the val images should be excluded from the list of training samples.

+ Download [**FSS-1000**](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI). 

##### To get voc_sbd_merge_noduplicate.txt:
+ We first merge the original VOC (voc_original_train.txt) and SBD ([**sbd_data.txt**](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt)) training data. 
+ [**Important**] sbd_data.txt does not overlap with the PASCALVOC 2012 validation data.
+ The merged list (voc_sbd_merge.txt) is then processed by the script (duplicate_removal.py) to remove the duplicate images and labels.

### Download Pretrained Models
+ Please download the pretrained models.
+ Update the config file by speficifying the target **split** and **path** (`weights`) for loading the checkpoint.
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.



## Train model

Execute this command at the root directory: 

train Difussion model once

​	set config/dataset/pascal_split0_resnet50.yaml epoch=1

```
# pascal
CUDA_VISIBLE_DEVICES=0 nohup python trainddpm.py --config=config/pascal/pascal_split0_resnet50.yaml >run0.out 2>&1 &
```

test Difussion model and give rough mask

```
# pascal
CUDA_VISIBLE_DEVICES=0 nohup python testddpm.py --config=config/pascal/pascal_split0_resnet50.yaml >run1.out 2>&1 &
```

#### train IRMNet

set config/dataset/pascal_split0_resnet50.yaml epoch=100

```
# pascal
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config=config/pascal/pascal_split0_resnet50.yaml >run2.out 2>&1 &
```

## Test model

test IRMNet
```
# pascal
CUDA_VISIBLE_DEVICES=0 nohup python test.py --config=config/pascal/pascal_split0_resnet50.yaml >run3.out 2>&1 &
```
