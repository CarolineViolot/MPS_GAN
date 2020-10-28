#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# SAMPLE COVARIANCE FUNCTION ON 2D IMAGE

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

#%% SAMPLE COV FUNC DEFINITION
def cov_func(im,dlag,n,toplag,lagseq="lin"): # takes univariate image/time series
    sx=np.shape(im)[0]
    x=np.random.randint(0,sx,n)
    X=np.repeat(x[:,None],n,axis=1)
    dX=X-np.transpose(X)
    del X
    if len(np.shape(im))==2:
        sy=np.shape(im)[1]
        y=np.random.randint(0,sy,n)
        Y=np.repeat(y[:,None],n,axis=1)
        dY=Y-np.transpose(Y)
        del Y
        D=np.sqrt(dX*dX+dY*dY)
        del dX,dY
        z=im[y,x]
    else:
        D=np.copy(dX)
        del dX
        z=im[x]
        sy=sx
    Z=np.repeat(z[:,None],n,axis=1)
    print("Z.shape : ", Z.shape)
    mZ=np.nanmean(Z)
    
    dZ=(Z-mZ)*(np.transpose(Z)-mZ)
    del Z
    if lagseq=="lin":
        lags=np.arange(0,toplag,dlag)
        lags=np.append(lags,toplag)
        lags=np.unique(lags)
    else:
        lags=np.round(np.logspace(0,np.log10(toplag),num=np.int(toplag/dlag)))
        lags=np.append(lags,0)
        lags=np.append(lags,toplag)
        lags=np.unique(lags)
    finlags=np.arange(0,toplag+1) # final 1-spaced lags
    v=np.empty_like(finlags)*np.nan
    for i in range(0,len(lags)):
        dZtmp=dZ[np.logical_and(D>lags[i]-dlag/2,D<=lags[i]+dlag/2)] # select couples for i-th lag
        v[finlags==lags[i]]=np.nanmean(dZtmp)
    
    f = interpolate.interp1d(finlags[np.isfinite(v)], v[np.isfinite(v)],kind="cubic")
    v[np.isnan(v)]=f(finlags[np.isnan(v)])
    return finlags, v # outplut lags-cov func value couples

#%% IMPORT TRAINING IMAGE
ti_r = np.array(ImageOps.grayscale(Image.open('../Datasets/Gaussian64x64/Images/image1.png')))
ti_g = np.array(ImageOps.grayscale(Image.open('../GeneratedImages/GAN/Gaussian64x64/NICE/fake_e_3_560.png')))

plt.figure()
plt.imshow(ti_r)
plt.title("Real Image")

plt.figure()
plt.imshow(ti_g)
plt.title("Generated Image")
#%% COMPUTE SAMPLE COVARIANCE FUNCTION
dlag=1 # compute a value every 10-pixel lag (interpolated in between), increase for speed, decrease for precision
toplag=40 # last value to compute, usually not larger than half the image
npoints=1000 # number of random points to use in the image, decrease for speed, increase for stability
#lags,c=cov_func(ti,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples

#plot

plt.figure()
for i in range(0,3):
    lags_r,c_r=cov_func(ti_r,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    lags_g,c_g=cov_func(ti_g,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    plt.plot(lags_r,c_r, 'b')
    plt.plot(lags_g, c_g, 'r')
    plt.xlabel('lag [pixels]')
    plt.ylabel('cov func value')
    plt.title('Image Covariance')

#%% IMPORT TRAINING IMAGE


ti_r = np.array(ImageOps.grayscale(Image.open('../Datasets/Gaussian64x64/Images/image1.png')))
ti_g = np.array(ImageOps.grayscale(Image.open('../GeneratedImages/GAN/Gaussian64x64/NICE/fake_e_3_920.png')))

plt.figure()
plt.imshow(ti_r)
plt.title("Real Image")

plt.figure()
plt.imshow(ti_g)
plt.title("Generated Image")
#%% COMPUTE SAMPLE COVARIANCE FUNCTION
dlag=1 # compute a value every 10-pixel lag (interpolated in between), increase for speed, decrease for precision
toplag=30 # last value to compute, usually not larger than half the image
npoints=1000 # number of random points to use in the image, decrease for speed, incrase for stability

plt.figure()
lags_r,c_r=cov_func(ti_r,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
lags_g,c_g=cov_func(ti_g,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
plt.plot(lags_r,c_r, 'b', label = 'real images')
plt.plot(lags_g, c_g, 'r', label = 'generated images')

datapath = '../GeneratedImages/MPS/Gaussian64x64/'
image_names = ['image1.png', 'image2.png', 'image3.png']    
for i in range(1,3):
    ti_g = np.array(ImageOps.grayscale(Image.open(datapath+image_names[i])))
    ti_r = np.array(ImageOps.grayscale(Image.open('../Datasets/Gaussian64x64/Images/image'+str(i)+'.png')))

    lags_r,c_r=cov_func(ti_r,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    lags_g,c_g=cov_func(ti_g,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    
    plt.plot(lags_r,c_r,'b')
    plt.plot(lags_g,c_g,'r')
    
    plt.legend()
    
    plt.xlabel('lag [pixels]')
    plt.ylabel('cov func value')



#%% STONE
#%% IMPORT TRAINING IMAGE
ti_g = np.array(ImageOps.grayscale(Image.open('../GeneratedImages/GAN/Stone/NICE/fake_e_4_480.png')))
ti_r = np.array(ImageOps.grayscale(Image.open('../Datasets/Stone/Images/sim_0.png')))

plt.figure()
plt.imshow(ti_r)
plt.title("Real Image")

plt.figure()
plt.imshow(ti_g)
plt.title("Generated Image")
#%% COMPUTE SAMPLE COVARIANCE FUNCTION

imagetype = 'Strebelle'
generativeMethod = 'MPS'

genImagepath = ''
images_names = []

if generativeMethod == 'MPS':
    genImagepath = '../GeneratedImages/MPS/'+imagetype+'/'
    images_names = ['image1.png','image2.png','image3.png']
elif generativeMethod == 'GAN':
    genImagepath = '../GeneratedImages/GAN/'+imagetype+'/NICE/'
    if imagetype == 'Gaussian64x64':
        images_names = ['fake_e_4_540.png', 'fake_e_4_560.png', 'fake_e_4_580.png']    
    if imagetype == 'Stone':
        images_names = ['fake_e_4_500.png', 'fake_e_4_520.png', 'fake_e_4_540.png']
    if imagetype == 'Strebelle':
        images_names = ['fake_e_1_8980.png', 'fake_e_1_8720.png', 'fake_e_1_8560.png']

dlag=10 # compute a value every 10-pixel lag (interpolated in between), increase for speed, decrease for precision
toplag=40 # last value to compute, usually not larger than half the image
npoints=500 # number of random points to use in the image, decrease for speed, incrase for stability

plt.figure()
lags_r,c_r=cov_func(ti_r,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
lags_g,c_g=cov_func(ti_g,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
plt.plot(lags_r,c_r, 'b', label = 'real images')
plt.plot(lags_g, c_g, 'r', label = 'generated images')
print(images_names)
    
for i in range(0,3):
    print(images_names[i])
    ti_g = np.array(ImageOps.grayscale(Image.open(genImagepath+images_names[i])))
    ti_r = np.array(ImageOps.grayscale(Image.open('../Datasets/'+imagetype+'/Images/sim_'+str(1)+'.png')))

    lags_r,c_r=cov_func(ti_r,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    lags_g,c_g=cov_func(ti_g,dlag,npoints,toplag,lagseq="lin") # gives (lag-covf) point couples
    
    plt.plot(lags_r,c_r, 'b')
    plt.plot(lags_g, c_g, 'r')
    
    plt.legend()
    
    plt.xlabel('lag [pixels]')
    plt.ylabel('cov func value')

