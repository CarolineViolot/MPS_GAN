#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:43:47 2020

@author: caroline
"""

import imageio
import os

images = []
for filename in  os.listdir("GeneratedImages/GAN"):
    images.append(imageio.imread(filename))
    imageio.mimsave('/GeneratedImages/GANGIF.gif', images)
    
    