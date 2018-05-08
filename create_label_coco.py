import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

train_img_dir="./coco_train2017/"
val_img_dir="./coco_val2017/"

coco=COCO("./coco_annotations/instances_train2014.json")

imgIds = coco.getImgIds()
img_heads = coco.loadImgs(imgIds)

classnum=90
labels=np.zeros([len(img_heads),classnum])
i=0
for img in img_heads:
  annIds=coco.getAnnIds(imgIds=img['id'])
  anns=coco.loadAnns(annIds)
  for ann in anns:
    labels[i][ann['category_id']-1]=labels[i][ann['category_id']-1]+1
  i=i+1

np.save("./coco_train_labels.npy",labels)
