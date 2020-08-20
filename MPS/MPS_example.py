#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#   SIMULATION A OF A BW IMAGE USING G2S (QS ALGORITHM)

from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
#import math
from g2s import run as g2s

#%% IMPORT TRAINING IMAGE
ti = np.array(Image.open('stone.png'))
plt.imshow(ti)

#%% ALGO PARAMETERS
serverAddress='localhost' #'tesla-k20c.gaia.unil.ch' # used server address
ti = ti # training image
di = np.empty_like(ti)*np.nan # empty simulation grid (destination image)
#di = np.empty(300,300)*np.nan # simulation grid of arbitrary size
dt = np.array([0]) # data type (numpy array, 1 element per variable): 0 = continuous, 1 = categorical
k = 3 # quantile threshold, k=3 means the 3 best pattern are taken for every pixel simulation
n = 200 # maximum number of neighbors considered to form the pattern (the n informed and closest)
#ki = np.ones([200,200]) # uniform kernel, it gives weights to the nieghbor composing the pattern
kernel=np.ones(shape=(201,201)); # gaussian kernel
kernel[101,101]=0;
kernle=ndimage.morphology.distance_transform_edt(kernel)
ki = np.power(2,kernel)

s = 100 # initial random seed
jb = 4 # number of parallel jobs
nr = 1 # number of realizations

#%% LAUNCH SIMULATION
# IMPORTANT: before launching the sim, start the server from the G2S folder: ~/lib/G2S/build/c++-build$ ./server

qssim=np.empty(np.hstack([np.shape(di),nr]))*np.nan   
for i in range(nr):
    s=s+1
    print("realization " + str(i))
    simout=g2s('-sa',serverAddress,'-a','qs','-ti',ti,'-di',di,'-dt',dt,'-ki',ki,'-k', k,'-n', n,'-s', s, '-j',jb);
    qssim[:,:,i]=simout[0]

#%% VISUALIZE RESULT
# ti vs sim images
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ti)
plt.title("training image")
plt.subplot(1,2,2)
plt.imshow(qssim[:,:,0])
plt.title("qs simulation")

# histogram
plt.figure()
plt.hist(qssim[:,:,0].ravel(),bins=50,alpha=.50,label="sim")
plt.hist(ti.ravel(),alpha=.50,bins=50,label="ti")
plt.legend()
plt.title("image histogram")
