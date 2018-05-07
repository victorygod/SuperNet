# SuperNet

A network who generates network from a small vector. Team member: Hantian Li, Yiming Ma.

## Introduction

When we are dealing with similar problem using neuron networks, we usually need to train a new network or fine-tune the original network. This involves of trainging millions of weights. However, for the most common situation, we found that the networks for similar problems are very similar, which means the actual variables in the transformation from the original network to target network are very few. So we came up with an idea that for a series of similar problems, we can generate the network from a small vector to deal with each of the problem. We did experiments on the cifar dataset. The result shows it works. 


## Related work

### cifar Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

Here are the classes in the dataset, as well as 10 random images from each:

![cifar.png](https://github.com/victorygod/SuperNet/blob/master/cifar.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

Here we use CIFAR-100 dataset, because our experiments are studying tasks on similar problems, which requires as many classes as possible.

## Algorithm

Our study of SuperNet is specified on classification tasks. Our network is sperated into two part: Generating network and Classifying network. 

The Generating network is a multi-task fully conncected network. It takes in a small vector as input. It generates every weights in the Classifying network.

