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

from ccpi.astra.operators import AstraProjector3DSimple, AstraProjectorSimple
from ccpi.plugins.regularisers import FGP_TV, SB_TV
from timeit import default_timer as timer
import astra
from ccpi.astra.processors import FBP

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

# Parameters for Acquisition Geometry
Horiz_det = int(np.sqrt(2)*N_size) # detector column count 
Vert_det = N_size # detector row count (vertical) 
angles_num = 100 # angles number
angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)

param = 10.5
# Cone geometry details
distance_source_origin = 300 * param  # [mm]
distance_origin_detector = 100 * param # [mm]
detector_pixel_size = 1.5 * param  # [mm]

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
                             dimension_labels=['vertical','angle','horizontal'])                       
    
    ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                            voxel_size_x = 1,
                            voxel_size_y = 1,
                            voxel_size_z = 1)
            
elif method == 'shifting':    
    
    
    ag = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                             angles = angles_rad, pixel_num_h=Horiz_det, 
                             pixel_num_v = Vert_det, 
                             pixel_size_h = 1, 
                             pixel_size_v = 1, 
                             dist_center_detector = 0, 
                             dist_source_center= (distance_source_origin + distance_origin_detector)/detector_pixel_size, 
                             dimension_labels=['vertical','angle','horizontal'])     
    
    ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                            voxel_size_x = 1,
                            voxel_size_y = 1,
                            voxel_size_z = 1)      
                      
elif method == 'with_unit':
    
    ag = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                             angles = angles_rad, pixel_num_h = Horiz_det, 
                             pixel_num_v = Vert_det, 
                             pixel_size_h = detector_pixel_size ,
                             pixel_size_v = detector_pixel_size ,
                             dist_center_detector = distance_origin_detector , 
                             dist_source_center= distance_source_origin , 
                             dimension_labels=['vertical','angle','horizontal'])                    

    
    ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                            voxel_size_x = detector_pixel_size / mag,
                            voxel_size_y = detector_pixel_size / mag,
                            voxel_size_z = detector_pixel_size / mag)          


filter_type = 'ram-lak'

X = ImageData(phantom_tm, geometry = ig)
Aop_cone = AstraProjector3DSimple(ig, ag)

sin_cone = Aop_cone.direct(X)

if method == 'with_unit':
    scaling =  (detector_pixel_size * ig.voxel_size_x/mag)**2
else:
    scaling = 1

fbp = FBP(ig, ag, filter_type = filter_type)
fbp.set_input(sin_cone)
fbp_fdk = fbp.get_output() /(scaling)

vol_geom_cone, proj_geom_cone = convert_geometry_to_astra(ig, ag)
          
rec_id = astra.data3d.create('-vol', vol_geom_cone)
sinogram_id = astra.data3d.create('-sino', proj_geom_cone, sin_cone.as_array())

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
    
rec = astra.data3d.get(rec_id) / (scaling)
astra.data3d.delete(rec_id)
astra.data3d.delete(sinogram_id)
            
astra.algorithm.delete(alg_id)

# assert fdk astra and ccpi-astra 
np.testing.assert_array_equal(fbp_fdk.as_array(), rec)

slice_ind = int(N_size/2)    

plt.figure(figsize = (10,30)) 
plt.subplot(131)
plt.imshow(fbp_fdk.as_array()[slice_ind,:,:])
plt.colorbar()
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(fbp_fdk.as_array()[:,slice_ind,:])
plt.colorbar()
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(fbp_fdk.as_array()[:,:,slice_ind])
plt.title('3D Phantom, sagittal view')
plt.colorbar()
plt.show()

plt.figure(figsize = (10,30)) 
plt.subplot(131)
plt.imshow(rec[slice_ind,:,:])
plt.colorbar()
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(rec[:,slice_ind,:])
plt.colorbar()
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(rec[:,:,slice_ind])
plt.title('3D Phantom, sagittal view')
plt.colorbar()
plt.show()
#

