#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:32:54 2019

@author: evangelos
"""

from ccpi.framework import DataProcessor, ImageData, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra
from ccpi.framework import ImageGeometry, ImageData, \
                            AcquisitionGeometry
                            
import os       
import tomophantom
from tomophantom import TomoP2D
from ccpi.astra.operators import AstraProjectorSimple 
import os
import numpy as np
import matplotlib.pyplot as plt

# For 2D objects we have CPU/GPU
# For 3D objects we have only GPU


# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 64 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

detectors =  N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

Aop = AstraProjectorSimple(ig, ag, 'cpu') 
sin = Aop.direct(data)

fbp = Aop.FBP(sin)

plt.imshow(fbp.as_array())
plt.show()

