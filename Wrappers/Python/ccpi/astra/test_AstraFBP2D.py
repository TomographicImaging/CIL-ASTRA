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

voxels = 0.0025

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                   voxel_size_x = voxels, voxel_size_y = voxels)
data = ImageData(phantom_2D)

detectors =  N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors, pixel_size_h = voxels)

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
print('There is a mismatch between CPU/GPU with the voxel size')
print('https://github.com/astra-toolbox/astra-toolbox/issues/38')
print('It is better to have image pixels with 1 and change detector pixel')
print('######################')

#%% Check astra FBP and ccpi-astra

filter_type = 'ram-lak'
               
dev = 'cpu'

if dev == 'cpu':    
    proj_id = astra.create_projector('line', proj_geom, vol_geom)
else:
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
    
sinogram_id, sinogram = astra.create_sino(data.as_array(), proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)

if dev =='cpu':
    cfg = astra.astra_dict('FBP')
    cfg['ProjectorId'] = proj_id
else:
    cfg = astra.astra_dict('FBP_CUDA')
    
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['FilterType'] = filter_type
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

rec = astra.data2d.get(rec_id)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

if dev == 'cpu':
    
    fbp_cpu = Aop_cpu.FBP(sin_cpu, filter_type)
    
    plt.imshow(fbp_cpu.as_array())
    plt.colorbar()
    plt.show()
    
    plt.imshow(rec)
    plt.colorbar()
    plt.show()
    
    np.testing.assert_array_equal(rec, fbp_cpu.as_array())
    
    
else:
    
    fbp_gpu = Aop_gpu.FBP(sin_gpu, filter_type)
    
    plt.imshow(fbp_gpu.as_array())
    plt.colorbar()
    plt.show()
    
    plt.imshow(rec)
    plt.colorbar()
    plt.show()
    
    plt.imshow(np.abs(rec - fbp_gpu.as_array()))
    plt.colorbar()
    plt.show()
    
    np.testing.assert_array_equal((ig.voxel_size_x**3)*rec, fbp_gpu.as_array())
    
    
#   (ig.voxel_size_x**3) * 
#%%

proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(data.as_array(), proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)

cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['FilterType'] = filter_type
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

rec_gpu = astra.data2d.get(rec_id)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

fbp_gpu = Aop_gpu.FBP(sin_gpu, filter_type)
#%%

#np.testing.assert_array_equal(sinogram, sin_gpu.as_array())  
np.testing.assert_array_equal(rec_gpu, fbp_gpu.as_array())  

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
      
          