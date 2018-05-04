# SuperNet

A network who generates network from a small vector. Team member: Hantian Li, Yiming Ma.

## Introduction

When we are dealing with similar problem using neuron networks, we usually need to train a new network or fine-tune the original network. This involves of changes of millions of weights. However, for the most common situation, we found that the networks for similar problems are very similar, which means the actually variable in this transformation from the original network to target network is very few. So we came up with an idea that for a series of similar problems, we can generate the network to deal with each of the problem from a small vector. We did experiments on the Cifar dataset. The result shows it works. 
