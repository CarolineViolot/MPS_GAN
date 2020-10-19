# Overview

This project aims to compare two state-of-the-art stochastic generative techniques to
simulate the heterogeneity observed in multi-spectral satellite imagery, widely
used to monitor Earth surface processes. Those two methods are : 

**1. Generative Adversarial Network (GAN)**, a recent family of neural network that is being widely applied in the fields of
image analysis an remote sensing. The fundamental GAN architecture consists of two neural networks playing against each other. One generates new images and the other discriminates these generated images against real images. The two neural networks iterate the generation while playing a minmax game where the discriminator gets better at classifying real and fake images and the generator gets better at creating images that fool the discriminator. After a usually intensive training phase, this strategy allows the generator to create realistic images that preserves complex features of the training images.

**2. Multiple-Point Statistical (MPS) resampling techniques**, which aims at generating realistic images by randomly sampling a training dataset and forming new data patterns. Simulated with an iterative workflow, the output images preserve complex spatial features and statistical relations that characterize natural processes, avoiding the formulation of complex probabilistic models.

# Report 

https://www.overleaf.com/read/qvmvpbcwgjzz
