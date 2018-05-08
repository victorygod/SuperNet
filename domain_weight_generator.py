import tensorflow as tf
import numpy as np
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import random
from skimage import io

# Notice that this graph directly link the generated weights to the task classifier for effiency of training
batch_size=1
hw=(224,224)
x=tf.placeholder(tf.float32,shape=[batch_size,hw[0],hw[1],3])
y=tf.placeholder(tf.float32)
dm=tf.placeholder(tf.float32,shape=[batch_size,365])

#
def supernet_brunch(inputs,innershapes=[128,128],outputshape=[3,3,128,128]):
  x=inputs
  for i in range(len(innershapes)):
    x=tf.layers.dropout(x,rate=0.2)
    x=tf.layers.dense(x,innershapes[i],use_bias=False)
    x=tf.layers.batch_normalization(x)
    x=tf.nn.relu(x)
  outdim=1
  for t in outputshape:
    outdim=outdim*t
  x=tf.layers.dense(x,outdim,activation=tf.nn.sigmoid,use_bias=True)
  x=tf.reshape(x,outputshape)
  return x

# We want to make the task classifier as tight as possible; here we temporarily set it to be five layers of convolution:
# [7,7,64] [5,5,128] [3,3,128] [3,3,256] [3,3,256] 
# fc: [1024] [classnum]
classnum=100

kernel1=supernet_brunch(dm,innershapes=[16,16],outputshape=[7,7,3,64])
kernel2=supernet_brunch(dm,innershapes=[64,64],outputshape=[5,5,64,128])
kernel3=supernet_brunch(dm,innershapes=[32,64],outputshape=[3,3,128,128])
kernel4=supernet_brunch(dm,innershapes=[32,64],outputshape=[3,3,128,256])
kernel5=supernet_brunch(dm,innershapes=[64,64],outputshape=[3,3,256,256])
W1=supernet_brunch(dm,innershapes=[32,32,64],outputshape=[int(hw[0]/32*hw[1]/32*256),1024])
W2=supernet_brunch(dm,innershapes=[32,32],outputshape=[1024,classnum])

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
  
  conv5=tf.reshape(conv5,shape=[np.shape(conv5)[0],-1])
  fc1=tf.matmul(conv5,fcs[0])
  fc1=tf.layers.batch_normalization(fc1,trainable=False)
  fc1=tf.layers.dropout(fc1,rate=0.5)
  fc1=tf.nn.relu(fc1)
  fc2=tf.matmul(fc1,fcs[1])
  fc2=tf.layers.batch_normalization(fc2,trainable=False)
  fc2=tf.nn.relu(fc2)
  return fc2

y_pred=task_classifier(x,[kernel1,kernel2,kernel3,kernel4,kernel5],[W1,W2])

#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
loss=tf.reduce_mean(tf.nn.l2_loss(y-y_pred))

train_step=tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()

# prepare dataset

train_img_dir="./coco_train2017/"
val_img_dir="./coco_val2017/"

coco=COCO("./coco_annotations/instances_train2014.json")

imgIds = coco.getImgIds()
img_heads = coco.loadImgs(imgIds)

max_epoch=10000

labels=np.load('./coco_train_labels.npy')
dm=np.load('./dm.npy')
index=np.arange(len(labels))

loss_avg=0
loss_count=0
test_iter=10
save_iter=5

save_path="./supernet_models/"
saver=tf.train.Saver(max_to_keep=1)

sess.run(init)
for e in max_epoch:
  #get image batch
  random.shuffle(index)
  loss_avg=0
  loss_count=0
  for index_i in index:
    img_feed=io.imread(train_img_dir+img_heads[index_i]['file_name'])
    label_feed=labels[index_i]
    dm_feed=dm[index_i]
    loss=sess.run(train_step,feed_dict={x:img_feed,y:label_feed,dm:dm_feed})
    print(loss)
    loss_avg=loss_avg+loss
  loss_avg=loss_avg/len(index)
  print("epoch_"+str(e)+": "+str(loss_avg))
  if (e+1)%save_iter==0:
    saver.save(save_path+"iter_"+str(e)+".ckpt")
  #if (e+1)%test_iter==0:
    #evaluation
