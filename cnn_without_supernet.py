import tensorflow as tf
import numpy as np
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import random
from skimage import io
import os
import cv2
# Notice that this graph directly link the generated weights to the task classifier for effiency of training
batch_size=1
hw=(224,224)
x=tf.placeholder(tf.float32,shape=[batch_size,hw[0],hw[0],3])
y=tf.placeholder(tf.float32)

# We want to make the task classifier as tight as possible; here we temporarily set it to be five layers of convolution:
# [7,7,64] [5,5,128] [3,3,128] [3,3,256] [3,3,256]
# fc: [1024] [classnum]
classnum=90


#[start building model]

def task_classifier(imgs):
  conv1=tf.layers.conv2d(imgs,filters=64,kernel_size=(7,7),strides=2,padding="same",use_bias=False,activation=tf.nn.relu,trainable=True)
  conv2=tf.layers.conv2d(conv1,filters=128,kernel_size=(5,5),strides=2,padding="same",use_bias=False,activation=tf.nn.relu,trainable=True)
  conv3=tf.layers.conv2d(conv2,filters=128,kernel_size=(3,3),strides=2,padding="same",use_bias=False,activation=tf.nn.relu,trainable=True)
  conv4=tf.layers.conv2d(conv3,filters=128,kernel_size=(3,3),strides=2,padding="same",use_bias=False,activation=tf.nn.relu,trainable=True)
  conv5=tf.layers.conv2d(conv4,filters=128,kernel_size=(3,3),strides=2,padding="same",use_bias=False,activation=tf.nn.relu,trainable=True)
  conv5=tf.reshape(conv5,shape=[tf.shape(conv5)[0],7*7*128])
  fc1=tf.layers.dense(conv5,units=512,use_bias=False,trainable=True)
  fc1=tf.layers.batch_normalization(fc1,trainable=False)
  fc1=tf.layers.dropout(fc1,rate=0.5)
  fc1=tf.nn.sigmoid(fc1)
  fc2=tf.layers.dense(fc1,units=classnum,use_bias=False,trainable=True)
  fc2=tf.layers.batch_normalization(fc2,trainable=False)
  fc2=tf.nn.sigmoid(fc2)
  return fc2

y_pred=task_classifier(x)

#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
loss=tf.reduce_mean(tf.nn.l2_loss(y-y_pred))

train_step=tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
#[end of building model]


# This function calculate this special task's accuracy
# here we expect the labels to be non-negative
def single_accuracy(pred,label):
  sum_value=0
  for i in range(len(pred)):
    abs_value=abs(pred[i]-label[i])
    if label[i]==0:
      if abs_value<0.03:
        sum_value=sum_value+1
      else:
        sum_value=sum_value+0
    elif abs_value>label[i]:
      sum_value=sum_value+0
    else:
      sum_value=sum_value+abs_value/label[i]
  return sum_value/(len(pred)+1e-10)

# This function only calculate the things that show up in an image (does not count cases when label==0)
def single_obj_accuracy(pred,label):
  sum_value=0
  counter=0
  for i in range(len(pred)):
    abs_value=abs(pred[i]-label[i])
    if label[i]!=0:
      if abs_value>label[i]:
        sum_value=sum_value+0
      else:
        sum_value=sum_value+abs_value/label[i]
      counter=counter+1
  if counter==0:
    return 0
  else:
    return sum_value/counter

#

# prepare dataset

train_img_dir="./coco_train2017/"
val_img_dir="./coco_val2017/"


# This part is for train use
coco=COCO("./coco_annotations/instances_train2014.json")
imgIds = coco.getImgIds()
img_heads = coco.loadImgs(imgIds)
labels=np.load('./norm_train_labels.npy')
index=np.arange(len(labels))
#

# This part is preparing data for test use
coco_val=COCO("./coco_annotations/instances_val2014.json")
imgIds_val=coco_val.getImgIds()
img_heads_val=coco_val.loadImgs(imgIds_val)
labels_val=np.load('./norm_val_labels.npy')
index_val=np.arange(len(labels_val))
#


max_epoch=10000
loss_avg=0
loss_count=0
test_iter=100
save_iter=100

# number of test imgs
test_number=300


save_path="./supernet_models/"
saver=tf.train.Saver(max_to_keep=1)

sess.run(init)

restore_mark=False
restore_iter=1
if restore_mark:
  saver.restore(sess,save_path+"iter_"+str(restore_iter)+".ckpt")
else:
  restore_iter=1

print("OK")
for e in range(restore_iter-1,max_epoch):
  # start of test
  avg_accuracy=0
  avg_accuracy_obj=0
  avg_count=0
  if (e+1)%test_iter==0:
    for index_j in index_val:
      #print(img_heads_val[index_j]['file_name'][13:])
      if not os.access(val_img_dir+img_heads_val[index_j]['file_name'][13:], os.R_OK):
        continue
      img=io.imread(val_img_dir+img_heads_val[index_j]['file_name'][13:])
      img=cv2.resize(img,hw)
      if np.shape(img)[-1]==1:
        img=np.array(np.concatenate([img,img,img],axis=-1))
      elif np.shape(img)[-1]!=3:
        continue
      img_feed=np.array([(img).astype(np.float32)])
      label_feed=np.array([labels_val[index_j]])
      pred=sess.run(y_pred,feed_dict={x:img_feed,y:label_feed})
      print("pred"+str(pred[0]))
      print("sample_accuracy: "+str(single_accuracy(pred[0],label_feed[0])))
      print("obj_accuracy: "+str(single_obj_accuracy(pred[0],label_feed[0])))
      avg_accuracy=avg_accuracy+single_accuracy(pred[0],label_feed[0])
      avg_accuracy_obj=avg_accuracy_obj+single_obj_accuracy(pred[0],label_feed[0])
      avg_count=avg_count+1
    avg_accuracy=avg_accuracy/avg_count
    avg_accuracy_obj=avg_accuracy_obj/avg_count
    print("test accuracy: "+str(avg_accuracy))
    print("test obj_accuracy"+str(avg_accuracy_obj))
  # end of test
  index_samples=np.random.choice(index,size=len(index)//1000)
  loss_avg=0
  loss_count=0
  avg_accuracy=0
  avg_accuracy_obj=0
  for index_i in index_samples:
    #if not os.access(train_img_dir+img_heads[index_i]['file_name'][15:], os.R_OK):
    #  continue
    img=io.imread(train_img_dir+img_heads[index_i]['file_name'][15:])
    img=cv2.resize(img,hw)
    if np.shape(img)[-1]==1:
      img=np.array(np.concatenate([img,img,img],axis=-1))
    elif np.shape(img)[-1]!=3:
      continue
    img_feed=np.array([(img).astype(np.float32)])
    label_feed=np.array([labels[index_i]])
    sess.run(train_step,feed_dict={x:img_feed,y:label_feed})
    loss_value=sess.run(loss,feed_dict={x:img_feed,y:label_feed})
    #print(loss_value)
    loss_avg=loss_avg+loss_value
    loss_count=loss_count+1
  loss_avg=loss_avg/loss_count
  print("epoch_"+str(e+1)+": "+str(loss_avg))
  if (e+1)%save_iter==0:
    saver.save(sess,save_path+"iter_"+str(e+1)+".ckpt")

