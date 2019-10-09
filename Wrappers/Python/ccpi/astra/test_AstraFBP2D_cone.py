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

from ccpi.astra.processors import FBP

# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 256 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

#vox = [0.03, 0.1, 3, 10]
vox = [6]

for voxels in vox: 
    
    ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N,
                       voxel_size_x = voxels, voxel_size_y = voxels)
    data = ImageData(phantom_2D)
    
    detectors =  N
    angles1 = np.linspace(0, 2*np.pi, 180, dtype=np.float32)
    angles2 = np.linspace(0, -2*np.pi, 180, dtype=np.float32)
    
    distance_source_origin = 500 # mm
    distance_origin_detector = 250 # mm
    detector_pixel_size = 0.8 # mm
    
    mag = (distance_source_origin + distance_origin_detector) / distance_source_origin
    voxel_real = detector_pixel_size / mag
    
    ag1 = AcquisitionGeometry('cone','2D', angles1, 
                             detectors, 
                             pixel_size_h = voxels * mag ,
                             dist_source_center = distance_source_origin/voxel_real,
                             dist_center_detector = distance_origin_detector/voxel_real)
    
    ag2 = AcquisitionGeometry('cone','2D', angles2, 
                             detectors, 
                             pixel_size_h = voxels * mag ,
                             dist_source_center = distance_source_origin/voxel_real,
                             dist_center_detector = distance_origin_detector/voxel_real)    
    
    # create astra geometries
#    vol_geom, proj_geom = convert_geometry_to_astra(ig, ag)
    
    # check cpu/gpu output
    Aop_cpu1 = AstraProjectorSimple(ig, ag1, 'cpu') 
    Aop_cpu2 = AstraProjectorSimple(ig, ag2, 'cpu') 
    
    
    Aop_gpu1 = AstraProjectorSimple(ig, ag1, 'gpu') 
    Aop_gpu2 = AstraProjectorSimple(ig, ag2, 'gpu')     
    
    # sinogram for cpu/gpu
    sin_cpu1 = Aop_cpu1.direct(data)
    sin_gpu1 = Aop_gpu1.direct(data)
    sin_cpu2 = Aop_cpu2.direct(data)
    sin_gpu2 = Aop_gpu2.direct(data)    
    
    print('######################')
    print('Check sinogram in CPU/GPU')
    print('######################')
          
    plt.imshow(sin_cpu2.as_array())
    plt.colorbar()
    plt.show()
    
    plt.imshow(sin_gpu1.as_array())
    plt.colorbar()
    plt.show()
    
    plt.imshow(np.abs(sin_cpu2.as_array() - sin_gpu1.as_array()))
    plt.colorbar()
    plt.show()
    
#    print('######################')
#    print('Check adjoint in CPU/GPU')
#    print('######################')
#          
#    plt.imshow(Aop_cpu.adjoint(sin_cpu).as_array())
#    plt.colorbar()
#    plt.show()
#    
#    plt.imshow(Aop_gpu.adjoint(sin_gpu).as_array())
#    plt.colorbar()
#    plt.show()
  
#%%    
#    plt.imshow(np.abs(sin_cpu.as_array() - sin_gpu.as_array()))
#    plt.colorbar()
#    plt.show()    
    
#    
#    print('######################')
#    print('There is a mismatch between CPU/GPU with the voxel size')
#    print('https://github.com/astra-toolbox/astra-toolbox/issues/38')
#    print('######################')
    
    #%% Check astra FBP and ccpi-astra
    
    filter_type = 'ram-lak'
                   
    dev = 'gpu'
    
    if dev == 'cpu':    
        proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    else:
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
    
    if dev == 'cpu':
        sinogram_id = astra.data2d.create('-sino', proj_geom, sin_cpu.as_array())
    else:
        sinogram_id = astra.data2d.create('-sino', proj_geom, sin_gpu.as_array())
        
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
    print('######################')
    print('There is a mismatch between CPU/GPU with the voxel size')
    print('https://github.com/astra-toolbox/astra-toolbox/issues/38')
    print('######################')
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    
    if dev == 'cpu':
            
        fbp = FBP(ig, ag, filter_type = filter_type)
        fbp.set_input(sin_cpu)
        fbp_cpu = fbp.get_output()
        
    #    fbp_cpu = Aop_cpu.FBP(sin_cpu, filter_type)
        
    
        rec = rec / (ig.voxel_size_x**2) 
    
        plt.imshow(fbp_cpu.as_array())
        plt.colorbar()
        plt.show()
        
        plt.imshow(rec)
        plt.colorbar()
        plt.show()
        
        plt.imshow(np.abs(rec - fbp_cpu.as_array()))
        plt.colorbar()
        plt.show()
        
        
        np.testing.assert_array_equal(rec, fbp_cpu.as_array())
        
        
    else:
        
        fbp = FBP(ig, ag, filter_type = filter_type, device = 'gpu')
        fbp.set_input(sin_gpu)
        fbp_gpu = fbp.get_output()    
        
      
        plt.imshow(fbp_gpu.as_array())
        plt.colorbar()
        plt.show()
        
        plt.imshow(rec)
        plt.colorbar()
        plt.show()
        
        plt.imshow(np.abs(rec - fbp_gpu.as_array()))
        plt.colorbar()
        plt.show()
    
        np.testing.assert_array_almost_equal(rec, fbp_gpu.as_array())
    
    

#%%

