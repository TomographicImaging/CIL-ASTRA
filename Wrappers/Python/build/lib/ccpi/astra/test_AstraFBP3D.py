import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D


from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData


from ccpi.optimisation.algorithms import PDHG, FISTA, CGLS

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, FunctionOperatorComposition

from ccpi.astra.operators import AstraProjector3DSimple, AstraProjectorSimple
from ccpi.plugins.regularisers import FGP_TV, SB_TV
from timeit import default_timer as timer


# Load Shepp-Logan Tomophantom 3D
print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
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
#angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)

# Setup ImageGeometry and AcquisitionGeometry
ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size)

ag = AcquisitionGeometry(geom_type = 'parallel', dimension = '3D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                         pixel_num_v=Vert_det, dimension_labels=['vertical','angle','horizontal'])


ig2D = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size)

ag2D = AcquisitionGeometry(geom_type = 'parallel', dimension = '2D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                          dimension_labels=['angle','horizontal'])

Aop = AstraProjector3DSimple(ig, ag)

X = ImageData(phantom_tm, geometry = ig)
sin = Aop.direct(X)  
fbp = Aop.FBP(sin, 'ram-lak')
   
slice_ind = int(N_size/2)    
plt.figure(figsize = (10,30)) 
plt.subplot(131)
plt.imshow(fbp.as_array()[slice_ind,:,:])
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(fbp.as_array()[:,slice_ind,:])
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(fbp.as_array()[:,:,slice_ind])
plt.title('3D Phantom, sagittal view')
plt.show()

ig_cone = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size)

ag_cone = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                         pixel_num_v=Vert_det, 
                         dist_center_detector=N_size, 
                         dist_source_center=5*N_size, 
                         dimension_labels=['vertical','angle','horizontal'])
Aop_cone = AstraProjector3DSimple(ig_cone, ag_cone)

sin_cone = Aop_cone.direct(X)

plt.imshow(sin_cone.as_array()[32])
plt.show()
#%%
from ccpi.astra.processors import AstraFDK, AstraFilteredBackProjector

#ttt = AstraFilteredBackProjector(ig, ag, 'ram-lak')
fdk = Aop_cone.FDK(sin_cone, 'ram-lak')
plt.imshow(fdk.as_array()[32])
plt.show()
#%%
#from ccpi.astra.processors import AstraFilteredBackProjector, AstraFDK
#
#ttt = AstraFDK(ig_cone, ag_cone, 'ram-lak')

#%%

#import astra
#
#vol_geom = astra.create_vol_geom(N_size, N_size, N_size)
#
#proj_geom = astra.create_proj_geom('cone',  
#                                   1, 
#                                   1, 
#                                   Vert_det, 
#                                   Horiz_det, 
#                                   angles_rad, 
#                                   5*N_size, 
#                                   N_size);
#
#sinogram_id, sinogram = astra.create_sino3d_gpu(phantom_tm, proj_geom, vol_geom)
#
#plt.imshow(sinogram[32])
#plt.show()
#
#
#
##%%
#### Create a data object for the reconstruction
#rec_id = astra.data3d.create('-vol', vol_geom)
###
#### Set up the parameters for a reconstruction algorithm using the GPU
#cfg = astra.astra_dict('FDK_CUDA')
#cfg['ReconstructionDataId'] = rec_id
#cfg['ProjectionDataId'] = sinogram_id
#alg_id = astra.algorithm.create(cfg)
#astra.algorithm.run(alg_id)
#
#res = astra.data3d.get(rec_id)
#
#astra.algorithm.delete(alg_id)
#astra.data3d.delete(rec_id)
#astra.data3d.delete(sinogram_id)
#
#plt.imshow(res[32])
#plt.show()
#
##%%
#
#
#
