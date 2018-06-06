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
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
dm=tf.placeholder(tf.float32,shape=[batch_size,365])

#
def supernet_brunch(inputs,innershapes=[128,128],outputshape=[3,3,128,128]):
  dur=inputs
  for i in range(len(innershapes)):
    dur=tf.layers.dropout(dur,rate=0.2)
    dur=tf.layers.dense(dur,innershapes[i],use_bias=False)
    dur=tf.layers.batch_normalization(dur)
    dur=tf.nn.relu(dur)
  outdim=1
  for t in outputshape:
    outdim=outdim*t
  dur=tf.layers.dense(dur,outdim,activation=tf.nn.tanh,use_bias=True)
  dur=tf.reshape(dur,outputshape)
  return dur

# We want to make the task classifier as tight as possible; here we temporarily set it to be five layers of convolution:
# [7,7,64] [5,5,128] [3,3,128] [3,3,256] [3,3,256] 
# fc: [1024] [classnum]
classnum=90


#[start building model]
kernel1=supernet_brunch(dm,innershapes=[16,16],outputshape=[7,7,3,64])
kernel2=supernet_brunch(dm,innershapes=[32,32],outputshape=[5,5,64,128])
kernel3=supernet_brunch(dm,innershapes=[32,32],outputshape=[3,3,128,128])
kernel4=supernet_brunch(dm,innershapes=[32,32],outputshape=[3,3,128,128])
kernel5=supernet_brunch(dm,innershapes=[32,32],outputshape=[3,3,128,128])
W1=supernet_brunch(dm,innershapes=[32,32],outputshape=[int(hw[0]/32*hw[1]/32*128),512])
W2=supernet_brunch(dm,innershapes=[32,32],outputshape=[512,classnum])

def task_classifier(imgs,kernels,fcs):
  print(np.shape(imgs))
  print(np.shape(kernels[0]))
  conv1=tf.nn.convolution(imgs,kernels[0],padding="SAME",strides=[2,2])
  conv1=tf.layers.batch_normalization(conv1,trainable=False)
  conv1=tf.nn.relu(conv1)
  
  conv2=tf.nn.convolution(conv1,kernels[1],padding="SAME",strides=[2,2])
  conv2=tf.layers.batch_normalization(conv2,trainable=False)
  conv2=tf.nn.relu(conv2)
  
  conv3=tf.nn.convolution(conv2,kernels[2],padding="SAME",strides=[2,2])
  conv3=tf.layers.batch_normalization(conv3,trainable=False)
  conv3=tf.nn.relu(conv3)
  
  conv4=tf.nn.convolution(conv3,kernels[3],padding="SAME",strides=[2,2])
  conv4=tf.layers.batch_normalization(conv4,trainable=False)
  conv4=tf.nn.relu(conv4)
  
  conv5=tf.nn.convolution(conv4,kernels[4],padding="SAME",strides=[2,2])
  conv5=tf.layers.batch_normalization(conv5,trainable=False)
  conv5=tf.nn.relu(conv5)
  
  conv5=tf.reshape(conv5,shape=[tf.shape(conv5)[0],-1])
  fc1=tf.matmul(conv5,fcs[0])
  fc1=tf.layers.batch_normalization(fc1,trainable=False)
  fc1=tf.layers.dropout(fc1,rate=0.5)
  fc1=tf.nn.sigmoid(fc1)
  fc2=tf.matmul(fc1,fcs[1])
  fc2=tf.layers.batch_normalization(fc2,trainable=False)
  fc2=tf.nn.sigmoid(fc2)
  return fc2

y_pred=task_classifier(x,[kernel1,kernel2,kernel3,kernel4,kernel5],[W1,W2])

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
dm_file=np.load('./dm_train.npy')
index=np.arange(len(labels))
#

# This part is preparing data for test use
coco_val=COCO("./coco_annotations/instances_val2014.json")
imgIds_val=coco_val.getImgIds()
img_heads_val=coco_val.loadImgs(imgIds_val)
labels_val=np.load('./norm_val_labels.npy')
dm_file_val=np.load('./dm_val.npy')
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

restore_mark=True
restore_iter=200
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
      dm_feed=np.array([dm_file_val[index_j]])
      pred=sess.run(y_pred,feed_dict={x:img_feed,y:label_feed,dm:dm_feed})
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
    dm_feed=np.array([dm_file[index_i]])
    sess.run(train_step,feed_dict={x:img_feed,y:label_feed,dm:dm_feed})
    loss_value=sess.run(loss,feed_dict={x:img_feed,y:label_feed,dm:dm_feed})
    #print(loss_value)
    loss_avg=loss_avg+loss_value
    loss_count=loss_count+1
  loss_avg=loss_avg/loss_count
  print("epoch_"+str(e+1)+": "+str(loss_avg))
  if (e+1)%save_iter==0:
    saver.save(sess,save_path+"iter_"+str(e+1)+".ckpt")

