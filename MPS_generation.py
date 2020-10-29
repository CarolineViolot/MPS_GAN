#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:47:55 2020

@author: caroline
"""

#   SIMULATION A OF A BW IMAGE USING G2S (QS ALGORITHM)

from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
#import math
from g2s import run as g2s
import time

#%% IMPORT TRAINING IMAGE
#rezising is not necessary as the dataset images have normally already been resized to 64x64 pixels
ti = np.array(Image.open('Datasets/Strebelle/Images/sim_1.png').convert('L').resize((64,64)))
#ti = np.array(Image.open('stone.png'))
plt.imshow(ti)
print(ti.shape)
#%% ALGO PARAMETERS
serverAddress='localhost' #'tesla-k20c.gaia.unil.ch' # used server address
ti = ti # training image
di = np.empty_like(ti)*np.nan # empty simulation grid (destination image)
#di = np.empty(300,300)*np.nan # simulation grid of arbitrary size
dt = np.array([0]) # data type (numpy array, 1 element per variable): 0 = continuous, 1 = categorical
k = 3 # quantile threshold, k=3 means the 3 best pattern are taken for every pixel simulation
n = 10 # maximum number of neighbors considered to form the pattern (the n informed and closest)
#ki = np.ones([200,200]) # uniform kernel, it gives weights to the nieghbor composing the pattern
kernel=np.ones(shape=(201,201)); # gaussian kernel
kernel[101,101]=0;
kernle=ndimage.morphology.distance_transform_edt(kernel)
ki = np.power(2,kernel)

s = 100 # initial random seed
jb = 4 # number of parallel jobs
nr = 100 # number of realizations

#%% LAUNCH SIMULATION
# IMPORTANT: before launching the sim, start the server from the G2S folder: ~/lib/G2S/build/c++-build$ ./server

qssim=np.empty(np.hstack([np.shape(di),nr]))*np.nan   
plt.figure(figsize=(4,4))
t = time.time()
for i in range(nr):
    s=s+1
    print("realization " + str(i))
    simout=g2s('-sa',serverAddress,'-a','qs','-ti',ti,'-di',di,'-dt',dt,'-ki',ki,'-k', k,'-n', n,'-s', s, '-j',jb);

    """
    qssim[:,:,i]=simout[0]
    #plt.subplot(4,4,i+1)
    plt.figure()
    plt.imshow(simout[0], cmap='gray')
    plt.axis('off')
    plt.tight_layout()    
    #plt.savefig('GeneratedImages/MPS/gaussian'+str(i)+'.png')
    """
print("time needed : " + str(time.time() - t))
#plt.tight_layout()    
#plt.savefig('generated_images/MPS/QS28.png')

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
#%%
#plt.imsave('generated_images/MPS/qs_simulation1.png', qssim[:,:,0], cmap='gray')
for i in range (0,10):
    arr = qssim[:,:,i]
    dpi = 96
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr,cmap='gray')
      
    plt.savefig("GeneratedImages/MPS/Strebelle/image"+ str(i) +".png", dpi=dpi)
            
