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
from mpl_toolkits.axes_grid1 import make_axes_locatable                 
import os       
import tomophantom
from tomophantom import TomoP3D
from ccpi.astra.operators import AstraProjector3DMC

import numpy as np
import matplotlib.pyplot as plt

from ccpi.astra.processors import FBP


model = 13 # select a model number from the library
N = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N, path_library3D)

num_chan = 5
phantom3DMC = np.tile(phantom_tm, [num_chan, 1, 1, 1])

#%%
angles = np.linspace(0, np.pi, 180)
#vox = [0.03, 0.1, 3, 10]
vox = [10]

for voxels in vox: 
    
    ig = ImageGeometry(N, N, N, channels = num_chan)
    
    data = ImageData(phantom3DMC)
    
    ag = AcquisitionGeometry('parallel','3D', angles, 
                             pixel_num_h = N, 
                             pixel_num_v = N, 
                             channels = num_chan,
                             dimension_labels = ['channel','vertical','angle','horizontal'])

    A3DMC = AstraProjector3DMC(ig, ag)
    # create astra geometries
    vol_geom, proj_geom = convert_geometry_to_astra(ig, ag)
    
    # check cpu/gpu output
    Aop_gpu = AstraProjector3DMC(ig, ag) 
    sin_gpu = Aop_gpu.direct(data)
    
    print('######################')
    print('Check sinogram in CPU/GPU')
    print('######################')
          
    plt.imshow(sin_gpu.as_array()[-1,32])
    plt.colorbar()
    plt.show()
    
##%%
 
filter_type = 'ram-lak'
#    
fbp = FBP(ig, ag, filter_type = filter_type)
fbp.set_input(sin_gpu)
fbp_gpu = fbp.get_output()  

#%%
plt.imshow(fbp_gpu.as_array()[4,32])
plt.colorbar()
plt.show()

#
fig, axs = plt.subplots(1, 3, figsize = (8,5))
cmap = 'viridis'

show_chan = [1,3,4]
tmp = fbp_gpu.as_array()
show_slices = [int(i/2) for i in  tmp.shape[1:]]

im1 = axs[0].imshow(tmp[show_chan[0],show_slices[0],:,:], cmap=cmap)
divider = make_axes_locatable(axs[0]) 
cax1 = divider.append_axes("right", size="5%", pad=0.1)      
fig.colorbar(im1, ax=axs[0], cax = cax1)   
    
im2 = axs[1].imshow(tmp[show_chan[1],:,show_slices[1],:], cmap=cmap)
divider = make_axes_locatable(axs[1])  
cax1 = divider.append_axes("right", size="5%", pad=0.1)       
fig.colorbar(im2, ax=axs[1], cax = cax1)   

im3 = axs[2].imshow(tmp[show_chan[2],:,:,show_slices[2]], cmap=cmap)
divider = make_axes_locatable(axs[2])  
cax1 = divider.append_axes("right", size="5%", pad=0.1) 
fig.colorbar(im3, ax=axs[2], cax = cax1)       
    
plt.tight_layout(h_pad=1)
#
## truth
#fig, axs = plt.subplots(1, 3, figsize = (8,5))
#tmp = phantom_2Dt
#
#im1 = axs[0].imshow(tmp[show_slices[0],:,:], cmap=cmap)
#divider = make_axes_locatable(axs[0]) 
#cax1 = divider.append_axes("right", size="5%", pad=0.1)      
#fig.colorbar(im1, ax=axs[0], cax = cax1)   
#    
#im2 = axs[1].imshow(tmp[show_slices[1],:,:], cmap=cmap)
#divider = make_axes_locatable(axs[1])  
#cax1 = divider.append_axes("right", size="5%", pad=0.1)       
#fig.colorbar(im2, ax=axs[1], cax = cax1)   
#
#im3 = axs[2].imshow(tmp[show_slices[2],:,:], cmap=cmap)
#divider = make_axes_locatable(axs[2])  
#cax1 = divider.append_axes("right", size="5%", pad=0.1) 
#fig.colorbar(im3, ax=axs[2], cax = cax1)       
#    
#plt.tight_layout(h_pad=1)
#
#
#  
##    #%% Check astra FBP and ccpi-astra
##    
##    filter_type = 'ram-lak'
##                   
##    dev = 'gpu'
##    
##    if dev == 'cpu':    
##        proj_id = astra.create_projector('line', proj_geom, vol_geom)
##    else:
##        proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
##        
##    sinogram_id, sinogram = astra.create_sino(data.as_array(), proj_id)
##    rec_id = astra.data2d.create('-vol', vol_geom)
##    
##    if dev =='cpu':
##        cfg = astra.astra_dict('FBP')
##        cfg['ProjectorId'] = proj_id
##    else:
##        cfg = astra.astra_dict('FBP_CUDA')
##        
##    cfg['ReconstructionDataId'] = rec_id
##    cfg['ProjectionDataId'] = sinogram_id
##    cfg['FilterType'] = filter_type
##    alg_id = astra.algorithm.create(cfg)
##    astra.algorithm.run(alg_id)
##    
##    rec = astra.data2d.get(rec_id) 
##    
##    astra.algorithm.delete(alg_id)
##    astra.data2d.delete(rec_id)
##    astra.data2d.delete(sinogram_id)
##    astra.projector.delete(proj_id)
##    
##    if dev == 'cpu':
##            
##        fbp = FBP(ig, ag, filter_type = filter_type)
##        fbp.set_input(sin_cpu)
##        fbp_cpu = fbp.get_output()
##        
##    #    fbp_cpu = Aop_cpu.FBP(sin_cpu, filter_type)
##        
##    
##        rec = rec / (ig.voxel_size_x**2) 
##    
##        plt.imshow(fbp_cpu.as_array())
##        plt.colorbar()
##        plt.show()
##        
##        plt.imshow(rec)
##        plt.colorbar()
##        plt.show()
##        
##        plt.imshow(np.abs(rec - fbp_cpu.as_array()))
##        plt.colorbar()
##        plt.show()
##        
##        
##        np.testing.assert_array_equal(rec, fbp_cpu.as_array())
##        
##        
##    else:
##        
##        fbp = FBP(ig, ag, filter_type = filter_type, device = 'gpu')
##        fbp.set_input(sin_gpu)
##        fbp_gpu = fbp.get_output()    
##        
##    #    fbp_gpu = Aop_gpu.FBP(sin_gpu, filter_type)
##        
##
##        
##        plt.imshow(fbp_gpu.as_array())
##        plt.colorbar()
##        plt.show()
##        
##        plt.imshow(rec)
##        plt.colorbar()
##        plt.show()
##        
##        plt.imshow(np.abs(rec - fbp_gpu.as_array()))
##        plt.colorbar()
##        plt.show()
##    
##        np.testing.assert_array_almost_equal(rec, fbp_gpu.as_array())
##    
##    
##
###%%
##
