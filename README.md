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

## Algorithm (Network Architecture)

Our study of SuperNet is specified on classification tasks. Our network is sperated into two part: Generating network and Classifying network. 

![superNet](https://github.com/victorygod/SuperNet/blob/master/supernet.png)

The Generating network is a multi-task fully conncected network. It takes in a small vector as input. It generates every weights in the Classifying network. The Classifying network takes in the input image. In every layer of the Classifying network, it multiply the data from last layer with the output of the Generating network. As the Generating network is multi-task, it generates different weights for every layer of the Classifying network. 

Since the operations here are all by multiplication, while we are doing backpropagation, the loss from the Classifying network will be finally propagated to the Generating network. So the only trainable weights are in Generating network. 

And there is a small redundant operation at the end of the Generating network, we add a tanh function. The reason we did this is this operation will restrict the generated weights in the Classifying network never exploded up. It is redundant, which means we can make the output linear. However, it is hard to train.

## Implementation and Analysis (Experiments)

We did multiple experienments to explore the character of the SuperNet.

### Experienment 1

The first experienment is to prove that we could get a new network by only changing the input vector of the Generating network. Here we call the input vector of the Generating network the vector z. 

Define a task i, Ti, as a binary classification task between class i and class 100 in CIFAR-100. Thus there are 99 tasks, which represented as T1~T99. For each task the ouput is a binary value, 1 represents the input image belongs to calss 100 and 0 to class i. The input of each task is restricted within those images belong to class 100 and class i.

As we have got 99 similar tasks, we can use T1~T98 to train a Generating network. We give an unique value of z for each of the task. In this experiment, vector z is 100-dimenson vector and we use the one-hot value as the initial value of z. We call this process Training phase. Then we freeze the network, using T99 to test whether could we get a suitable network only by changing the vector z. We call this process Testing phase.

The loss function we use is sigmoid cross entropy. 

The result is by only changing the vector z, the accuracy and loss are both decreasing to the same level of by changing the whole network.

### Experienment 2

Experienment 1 is not perfect because the Generating network may have already learned a general network. So we don't have to train a vector z in Testing phase to get a new network. To prove we do get a new network, we re-design our network and make it must resort to vector z. So in experienment 2, for the first 50 tasks, we make the network to classify between class i and class i+50 for each i in 0~49. And for the next 50 tasks, we make the network to classify the same classes but label them in the reverse order. For instance, in T1, the network will classify class 1 as label 0 and class 51 as label 1. At the mean time, in T51, the network will classify class 1 as label 1 and class 51 as label 0. One single network cannot do both of the above two tasks, so it has to resort to vector z.

The result of experienment 2 shows the network also converged, which means the network actually resort to vector z to get a new network.

### Experienment 3
This time we try out on a little more realistic and diffcult task: counting objects of every class for an input image on MS coco dataset. We use places as the domains. By running a fully trained place classifier trained on "MIT places2" dataset we get the place prediction for every image in coco. These predictions are not ensured to be accurate since places paper says their top5
 accuracy is 88% or so. There are 365 place classes so we just put the 365 dimensional vector at the input of our supernet, making only the supernet part trainable:
 
## Conclusion

The idea of SuperNet is that the weights of networks for similar tasks are also similar. And a new network could be generated from a small vector. This vector is a compress representation of this new network in the domain of such similar network. We designed experienments to prove that we can get a new network by only training a small vector. The meaning of SuperNet is that in transfer learning, if we use SuperNet to fit target domain, the trainable weights will be very few and the flexibility will be guaranteed.
