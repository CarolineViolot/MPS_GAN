#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:43:47 2020

@author: caroline
"""

import imageio
import os

with imageio.get_writer('GeneratedImages/GANGIFStrebelle.gif', mode='I', fps = 10) as writer:
    files = sorted(os.listdir("GeneratedImages/GAN/Strebelle/"))
    
    for e in range (0, 1        ):
        for i in range (0, 8980, 20):#0:20: filename in files:
            
            image = imageio.imread("GeneratedImages/GAN/Strebelle/fake_e_"+str(e)+"_"+str(i)+".png")
            writer.append_data(image)

