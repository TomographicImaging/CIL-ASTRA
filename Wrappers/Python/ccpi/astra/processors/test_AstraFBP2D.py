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
N = 256 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

detectors =  N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

# check cpu/gpu output
Aop_cpu = AstraProjectorSimple(ig, ag, 'cpu') 
Aop_gpu = AstraProjectorSimple(ig, ag, 'gpu') 

sin_cpu = Aop_cpu.direct(data)
sin_gpu = Aop_gpu.direct(data)

print('######################')
print('Check sinogram in CPU/GPU')
print('######################')
      
plt.imshow(sin_cpu.as_array())
plt.colorbar()
plt.show()

plt.imshow(sin_gpu.as_array())
plt.colorbar()
plt.show()

plt.imshow(np.abs(sin_cpu.as_array() - sin_gpu.as_array()))
plt.colorbar()
plt.show()

#%%

print('######################')
print('Compare FBP for cpu')
print('######################')
      
filt1 = 'cosine'
filt2 = 'lanczos'

fbp_cpu1 = Aop_cpu.FBP(sin_cpu, filt1)
fbp_cpu2 = Aop_cpu.FBP(sin_cpu, filt2)

plt.imshow(fbp_cpu1.as_array())
plt.colorbar()
plt.show()

plt.imshow(fbp_cpu2.as_array())
plt.colorbar()
plt.show()

plt.imshow(np.abs(fbp_cpu1.as_array() - fbp_cpu2.as_array()))
plt.colorbar()
plt.show()

print('######################')
print('Different filters produce the same result')
print('######################')

#%%

print('######################')
print('Compare FBP for gpu')
print('######################')
      
filt1 = 'cosine'
filt2 = 'hamming'

fbp_gpu1 = Aop_gpu.FBP(sin_gpu, filt1)
fbp_gpu2 = Aop_gpu.FBP(sin_gpu, filt2)

plt.imshow(fbp_gpu1.as_array())
plt.colorbar()
plt.show()

plt.imshow(fbp_gpu2.as_array())
plt.colorbar()
plt.show()

plt.imshow(np.abs(fbp_gpu1.as_array() -fbp_gpu2.as_array()))
plt.colorbar()
plt.show()
      
      
#%%      

#print('######################')
#print('Compare FBP for cpu/gpu for ram-lak. Here it works')
#print('######################')
#      
#filt = 'ram-lak'
#fbp_cpu = Aop_cpu.FBP(sin_cpu, filt)
#fbp_gpu = Aop_gpu.FBP(sin_gpu, filt)
#
#plt.imshow(fbp_cpu.as_array())
#plt.colorbar()
#plt.show()
#
#plt.imshow(fbp_gpu.as_array())
#plt.colorbar()
#plt.show()
#
#plt.imshow(np.abs(fbp_cpu.as_array() -fbp_gpu.as_array()))
#plt.colorbar()
#plt.show()
#
##%%
#
#filt1 = 'hamming'
##filt2 = 'ram-lak'
##
##fbp_cpu1 = Aop_cpu.FBP(sin_cpu, filt1)
##fbp_cpu2 = Aop_cpu.FBP(sin_cpu, filt2)
##
##plt.imshow(fbp_cpu1.as_array())
##plt.colorbar()
##plt.show()
##
##plt.imshow(fbp_cpu2.as_array())
##plt.colorbar()
##plt.show()
##
##plt.imshow(np.abs(fbp_cpu1.as_array() - fbp_cpu2.as_array()))
##plt.colorbar()
##plt.show()
##
##
##fbp_gpu1 = Aop_gpu.FBP(sin_gpu, filt1)
##fbp_gpu2 = Aop_gpu.FBP(sin_gpu, filt2)
##
##plt.imshow(fbp_gpu1.as_array())
##plt.colorbar()
##plt.show()
##
##plt.imshow(fbp_gpu2.as_array())
##plt.colorbar()
##plt.show()
##
##plt.imshow(np.abs(fbp_gpu1.as_array() - fbp_gpu2.as_array()))
##plt.colorbar()
##plt.show()
##
###plt.imshow(fbp_gpu.as_array())
###plt.colorbar()
###plt.show()
##
##
##
###plt.imshow(Aop_cpu.adjoint(sin_cpu).as_array())
###plt.colorbar()
###plt.show()
###
###plt.imshow(Aop_gpu.adjoint(sin_gpu).as_array())
###plt.colorbar()
###plt.show()
##
##
