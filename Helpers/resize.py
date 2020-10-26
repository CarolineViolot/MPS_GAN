#!/usr/bin/python
from PIL import Image
import os, sys

path = "../GeneratedImages/MPS/Gaussian64x64/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()
