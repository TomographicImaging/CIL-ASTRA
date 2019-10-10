#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:25:54 2019

@author: vaggelis
"""

import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from ccpi.astra.utils import convert_geometry_to_astra
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.algorithms import PDHG, FISTA, CGLS

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, FunctionOperatorComposition

from ccpi.astra.operators import AstraProjector3DMC
from ccpi.plugins.regularisers import FGP_TV, SB_TV
from timeit import default_timer as timer
import astra
from ccpi.astra.processors import FBP
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable 

phantom = 'lizard'

if phantom == 'shepp':

    num_chan = 5
    # Load Shepp-Logan Tomophantom 3D
    print ("Building 3D phantom using TomoPhantom software")
    tic=timeit.default_timer()
    model = 13 # select a model number from the library
    N_size = 256 # Define phantom dimensions using a scalar value (cubic phantom)
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
    #This will generate a N_size x N_size x N_size phantom (3D)
    phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
    toc=timeit.default_timer()
    Run_time = toc - tic
    print("Phantom has been built in {} seconds".format(Run_time))
    
    # Show Phantom in axial - coronal - sagittal view
    slice_ind = int(N_size/2)
    plt.figure(figsize = (10,30)) 
    plt.subplot(131)
    plt.imshow(phantom_tm[slice_ind,:,:],vmin=0, vmax=1)
    plt.title('3D Phantom, axial view')
    
    plt.subplot(132)
    plt.imshow(phantom_tm[:,slice_ind,:],vmin=0, vmax=1)
    plt.title('3D Phantom, coronal view')
    
    plt.subplot(133)
    plt.imshow(phantom_tm[:,:,slice_ind],vmin=0, vmax=1)
    plt.title('3D Phantom, sagittal view')
    plt.show()
    
    phantom3DMC = np.tile(phantom_tm, [num_chan, 1, 1, 1])    
    
    # Parameters for Acquisition Geometry
    Horiz_det = int(np.sqrt(2)*N_size) # detector column count 
    Vert_det = N_size # detector row count (vertical) 
    angles_num = 100 # angles number
    angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)
    
    param = 1
    # Cone geometry details
    distance_source_origin = 100 * param  # [mm]
    distance_origin_detector = 300 * param # [mm]
    detector_pixel_size = 0.05 * param  # [mm]    
    
elif phantom == 'lizard':

    pathname = '/media/newhd/shared/DataProcessed/Data_Ryan_Chris/reconstruction/'
    filename = 'sinogram_centered.h5'
    
    path = pathname + filename
    arrays = {}
    f = h5py.File(path)
    for k, v in f.items():
        arrays[k] = np.array(v)
    XX = arrays['SC']    
    phantom3DMC = XX[100:120] # small range of channels
    f.close()    
    
    # Parameters for Acquisition Geometry
    angles_num = phantom3DMC.shape[2] # angles number
    angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)
    
    # Cone geometry details
    distance_source_origin = 233 # [mm]
    distance_origin_detector = 245 # [mm]
    detector_pixel_size = 0.25   # [mm]
    
    Horiz_det = phantom3DMC.shape[3]
    Vert_det = phantom3DMC.shape[1]
    num_chan = phantom3DMC.shape[0]


# Compute magnification factor
mag = (distance_source_origin + distance_origin_detector) / distance_source_origin

# Real voxel size
voxel_real = detector_pixel_size / mag

method = 'with_unit'
    
if method == 'unitless':
  
    ag = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                             angles = angles_rad, pixel_num_h=Horiz_det, 
                             pixel_num_v = Vert_det, 
                             pixel_size_h = detector_pixel_size/voxel_real,
                             pixel_size_v = detector_pixel_size/voxel_real,
                             dist_center_detector = distance_origin_detector/voxel_real, 
                             dist_source_center= distance_source_origin/voxel_real, 
                             dimension_labels=['vertical','angle','horizontal'],
                             channels = num_chan)                       
    
    ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                            voxel_size_x = 1,
                            voxel_size_y = 1,
                            voxel_size_z = 1,
                            channels = num_chan)
            
elif method == 'shifting':    
    
    
    ag = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                             angles = angles_rad, pixel_num_h=Horiz_det, 
                             pixel_num_v = Vert_det, 
                             pixel_size_h = 1, 
                             pixel_size_v = 1, 
                             dist_center_detector = 0, 
                             dist_source_center= (distance_source_origin + distance_origin_detector)/detector_pixel_size, 
                             dimension_labels=['vertical','angle','horizontal'],
                             channels = num_chan)     
    
    ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                            voxel_size_x = 1,
                            voxel_size_y = 1,
                            voxel_size_z = 1,
                            channels = num_chan)      
                      
elif method == 'with_unit':
    
    ag = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                             angles = angles_rad, pixel_num_h = Horiz_det, 
                             pixel_num_v = Vert_det, 
                             pixel_size_h = detector_pixel_size ,
                             pixel_size_v = detector_pixel_size ,
                             dist_center_detector = distance_origin_detector , 
                             dist_source_center= distance_source_origin , 
                             dimension_labels=['channel','vertical','angle','horizontal'],
                             channels = num_chan, )                    

    
    ig = ImageGeometry(voxel_num_x=ag.pixel_num_h, voxel_num_y=ag.pixel_num_h, voxel_num_z=ag.pixel_num_h,
                            voxel_size_x = detector_pixel_size / mag,
                            voxel_size_y = detector_pixel_size / mag,
                            voxel_size_z = detector_pixel_size / mag,
                            channels = num_chan)          


filter_type = 'ram-lak'
Aop_cone = AstraProjector3DMC(ig, ag)

if phantom == 'shepp':
    X = ImageData(phantom3DMC, geometry = ig)
    sin_cone = Aop_cone.direct(X)
else:
    sin_cone = AcquisitionData(phantom3DMC, geometry = ag, dimension_labels = ['channel', 'vertical', 'angle', 'horizontal'] )
    

#%%

if method == 'with_unit':
    scaling =  (detector_pixel_size * ig.voxel_size_x/mag)**2
else:
    scaling = 1

fbp = FBP(ig, ag, filter_type = filter_type)
fbp.set_input(sin_cone)
fbp_fdk = fbp.get_output() /(scaling)

#%%

fig, axs = plt.subplots(1, 3, figsize = (12,7))
cmap = 'viridis'

show_chan = [1,3,4]
tmp = fbp_fdk.as_array()
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




