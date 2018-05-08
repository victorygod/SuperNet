# Resnet18 for place classification
# by:mym2358 
# Notice data are in NCHW format
# padding size need to be calculated for keeping shape of tensor(need to be fixed later)
#
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from skimage import io
import cv2


hw=(256,256)

class Block_normal(nn.Module):
  def __init__(self,in_channels,filternum=64,kernel_size=(3,3),use_conv_shortcut=False):
    super(Block_normal,self).__init__()
    self.pad_size=(1,1,1,1,0,0,0,0)
    #self.pad_size=((kernel_size[1]-1)/2,(kernel_size[1]-1)/2,(kernel_size[0]-1)/2,(kernel_size[0]-1)/2,0,0,0,0)
    self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=filternum,padding=0,kernel_size=kernel_size)
    self.BN1=nn.BatchNorm2d(filternum)
    self.conv2=nn.Conv2d(in_channels=filternum,out_channels=filternum,padding=0,kernel_size=kernel_size)
    self.BN2=nn.BatchNorm2d(filternum)
  def forward(self,x):
    x_pad=F.pad(x,self.pad_size)
    act1=F.relu(self.BN1(self.conv1(x_pad)))
    act1=F.pad(act1,self.pad_size)
    act2=F.relu(self.BN2(F.dropout(self.conv2(act1),0.1)))
    return act2+x
    
class Block_downsampling(nn.Module):
  def __init__(self,in_channels,filternum=64,kernel_size=(3,3),use_conv_shortcut=False,in_stride=2):
    super(Block_downsampling,self).__init__()
    self.pad_size1=(1,0,1,0,0,0,0,0)
    self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=filternum,kernel_size=kernel_size,stride=in_stride)
    self.BN1=nn.BatchNorm2d(filternum)
    self.pad_size2=(1,1,1,1,0,0,0,0)
    self.conv2=nn.Conv2d(in_channels=filternum,out_channels=filternum,kernel_size=kernel_size)
    self.BN2=nn.BatchNorm2d(filternum)   
    self.pad_size_shortcut=(0,0,0,0,0,0,0,0)
    self.conv_shortcut=nn.Conv2d(in_channels=in_channels,out_channels=filternum,kernel_size=(1,1),stride=in_stride)
  def forward(self,x):
    x_pad=F.pad(x,self.pad_size1)
    act1=F.relu(self.BN1(self.conv1(x_pad)))
    act1=F.pad(act1,self.pad_size2)
    act2=F.relu(self.BN2(F.dropout(self.conv2(act1),0.1)))
    shortcut_pad=F.pad(x,self.pad_size_shortcut)
    return act2+self.conv_shortcut(shortcut_pad)

class Pre_block(nn.Module):
  def __init__(self,in_channels,filternum=64,kernel_size=(7,7)):
    super(Pre_block,self).__init__()
    #self.pad_size_conv=((kernel_size[1]-1)/2,(kernel_size[1]-1)/2,(kernel_size[0]-1)/2,(kernel_size[0]-1)/2,0,0,0,0)
    self.pad_size_conv=(3,3,3,3,0,0,0,0)
    self.pad_size_pool=(0,0,0,0,0,0,0,0)
    self.conv=nn.Conv2d(in_channels=in_channels,out_channels=filternum,kernel_size=kernel_size)
    self.BN=nn.BatchNorm2d(filternum)
  def forward(self,x):
    x_pad=F.pad(x,self.pad_size_conv)
    dur=F.relu(self.BN(self.conv(x_pad)))
    dur_pad=F.pad(dur,self.pad_size_pool)
    output=F.max_pool2d(dur_pad,kernel_size=(2,2),stride=2)
    return output

class Resnet(nn.Module):
  def __init__(self,Pre_block,Block_downsampling,Block_normal,Block_normal_nums=[2,2,2,2],Block_normal_filters=[64,128,256,512],hw=(8,8),in_channels=3,classnum=365):
    super(Resnet,self).__init__()
    self.Preblock=Pre_block(in_channels=in_channels)
    self.build_layers(Block_normal,Block_normal_nums,Block_normal_filters)
    self.Downsamplingblocks=[]
    self.normblocknums=Block_normal_nums
    self.fc1=nn.Linear(hw[0]*hw[1]*Block_normal_filters[-1],1024)
    self.fc2=nn.Linear(1024,classnum)
    for i in range(len(Block_normal_nums)-1):
      #this only create for normal blocks after the first one
      self.Downsamplingblocks.append(Block_downsampling(in_channels=Block_normal_filters[i],filternum=Block_normal_filters[i+1]))
    
  def build_layers(self,Block_normal,Block_normal_nums,Block_normal_filters):
    self.Normblocks=[]
    for i in range(len(Block_normal_nums)):
      self.Normblocks.append([])
      for j in range(Block_normal_nums[i]):
        self.Normblocks[i].append(Block_normal(Block_normal_filters[i],Block_normal_filters[i]))

  def forward(self,x):
    x=self.Preblock(x)
    for i in range(len(self.normblocknums)):
      if i>0:
        x=self.Downsamplingblocks[i-1](x)
      for j in range(self.normblocknums[i]):
        x=self.Normblocks[i][j](x)
    x=F.avg_pool2d(x,kernel_size=2,stride=2)
    x=F.relu(self.fc1(F.dropout(x.view(x.size(0),-1),0.1)))
    x=F.relu(self.fc2(x))
    return x

  def show_modules(self):
    for m in self.modules():
      print(m)

Resnet18=Resnet(Pre_block,Block_downsampling,Block_normal,Block_normal_nums=[2,2,2,2],Block_normal_filters=[64,128,256,512],hw=(8,8),in_channels=3,classnum=365)
#The hw is the size before entering fully connected layers

loss_rule=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(Resnet18.parameters(), lr=1e-5, momentum=0.9)

max_batch=50000
test_iter=100
batch_size=2
in_channels=3
x=torch.Tensor(np.random.normal(size=[batch_size,hw[0],hw[1],3]))
y=torch.Tensor(np.random.normal(size=[batch_size,1]))


Resnet18.train()
#Resnet18.show_modules()
#loss_rule.cuda()
# Get data 

train_file=open('./places365_standard/train.txt','r')
val_file=open('./places365_standard/val.txt','r')
train_dir=train_file.readlines()
val_dir=val_file.readlines()
data_dir='./places365_standard/'

# Create category map: number->folder name
class_names=os.listdir(data_dir+'train')
file_names={}
for name in class_names:
  #print(name)
  file_names[name]=os.listdir(data_dir+'train/'+name+'/')


# To get data, random sample a label and then sample an image from that category, this is the same as random sampling over all data.
save_iter=50

restore_mark=1
restore_iter=1250

if restore_mark==1:
  Resnet18=torch.load("./resnet18/iter_"+str(restore_iter)+".pt")

for e in range(restore_iter,max_batch):
  # Every iter: 1.get batched data; 2.forward pass; 3.print(loss); 4.grad_zero; 5.backward pass; 6.update params
  x=[]
  #y=torch.Tensor(np.zeros([batch_size,len(class_names)]))
  labels=np.random.randint(0,len(class_names),size=batch_size)
  y=torch.LongTensor(labels)
  for b in range(batch_size):
    x.append(io.imread(data_dir+"train/"+class_names[labels[b]]+"/"+random.sample(file_names[class_names[labels[b]]],1)[0]))
    #y[b][labels[b]]=1.0
  x=np.swapaxes(x,2,3)
  x=np.swapaxes(x,1,2)
  x=torch.Tensor(x)
  #x=x.cuda()
  #Resnet18.cuda()
  y_pred=Resnet18(x)
  #y=y.cuda()
  loss=loss_rule(y_pred,y)
  print("epoch "+str(e)+": "+str(loss.item()))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if (e+1)%save_iter==0:
    torch.save(Resnet18,"./resnet18/iter_"+str(e+1)+".pt")




