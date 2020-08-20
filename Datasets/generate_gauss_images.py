#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#%%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'size'   : 14}
rc('font', **font)
from scipy.ndimage import gaussian_filter

#%% GENERATE GAUSSIAN FIELDS
# params
l=2000 # field size
s=50 # kernel variance 
r=np.random.rand(l,l) # generate random noise
C=gaussian_filter(r, sigma=s) # convolve with guassian kernel of sigma=s
plt.imshow(C)