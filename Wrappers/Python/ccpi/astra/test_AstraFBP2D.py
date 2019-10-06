#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:32:54 2019

@author: evangelos
"""
from ccpi.framework import ImageData, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra
from ccpi.framework import ImageGeometry, AcquisitionGeometry
                            
import os       
import tomophantom
from tomophantom import TomoP2D
from ccpi.astra.operators import AstraProjectorSimple 

import numpy as np
import matplotlib.pyplot as plt

# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 256 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

voxels = 0.25

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                   voxel_size_x = voxels, voxel_size_y=voxels)
data = ImageData(phantom_2D)

detectors =  N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

# create astra geometries
vol_geom, proj_geom = convert_geometry_to_astra(ig, ag)

# check cpu/gpu output
Aop_cpu = AstraProjectorSimple(ig, ag, 'cpu') 
Aop_gpu = AstraProjectorSimple(ig, ag, 'gpu') 

# sinogram for cpu/gpu
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

print('######################')
print('There is a mismatch between CPU/GPU: https://github.com/astra-toolbox/astra-toolbox/issues/38')
print('######################')

#%% Check astra FBP and ccpi-astra

devices = ['cpu','gpu']      
            
proj_id = astra.create_projector('line', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(data.as_array(), proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)

cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = proj_id
cfg['FilterType'] = 'ram-lak'

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

rec = astra.data2d.get(rec_id)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

fbp_cpu = Aop_cpu.FBP(sin_cpu, 'ram-lak')

np.testing.assert_array_equal(rec, fbp_cpu.as_array())
   
#%%   

print('######################')
print('Compare FBP for cpu')
print('######################')
      
filt1 = 'cosine'
filt2 = 'lanczos'

fbp_cpu1 = Aop_cpu.FBP(sin_cpu, filt1)
fbp_cpu2 = Aop_cpu.FBP(sin_cpu, filt2)



proj_id = astra.create_projector('strip', proj_geom, vol_geom)
# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)
sinogram_id, sinogram = astra.create_sino(phantom_2D, proj_id)


cfg = astra.astra_dict('SIRT')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = proj_id

# Available algorithms:
# ART, SART, SIRT, CGLS, FBP


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 20 iterations of the algorithm
# This will have a runtime in the order of 10 seconds.
astra.algorithm.run(alg_id, 20)


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
      
          