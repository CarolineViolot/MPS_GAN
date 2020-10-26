#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:35:53 2020

@author: caroline
"""

from PIL import Image
import os, sys
import matplotlib.pyplot as plt
import numpy as np
path = "../GeneratedImages/MPS/Stone/"
dirs = os.listdir( path )

def create_dataset_images(image_grid_rows, image_grid_columns):
    # Set image grid
    im = []
    stop = 0
    for item in dirs:#[image_grid_columns*image_grid_rows]:
        print(item)
        stop = stop + 1
        if stop > 20:
            break
        if os.path.isfile(path+item):
            
            im.append(np.array(Image.open(path+item)))
            print((np.array(Image.open(path+item))).shape)
           
    
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)
    
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(im[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.tight_layout()
    

create_dataset_images(2, 2)
