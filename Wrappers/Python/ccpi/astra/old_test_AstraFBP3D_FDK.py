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
angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)

voxels = 2.5

# Setup ImageGeometry and AcquisitionGeometry
ig = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                   voxel_size_x = voxels,
                   voxel_size_y = voxels,
                   voxel_size_z = voxels)

ag = AcquisitionGeometry(geom_type = 'parallel', dimension = '3D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                         pixel_num_v=Vert_det, 
                         pixel_size_h = voxels,
                         pixel_size_v = voxels,
                         dimension_labels=['vertical','angle','horizontal'])

ig2D = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size,
                   voxel_size_x = voxels,
                   voxel_size_y = voxels)

ag2D = AcquisitionGeometry(geom_type = 'parallel', dimension = '2D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                         pixel_size_h = voxels)


vol_geom2D, proj_geom2D = convert_geometry_to_astra(ig2D, ag2D)
vol_geom3D, proj_geom3D = convert_geometry_to_astra(ig, ag)

Aop = AstraProjector3DSimple(ig, ag)

X = ImageData(phantom_tm, geometry = ig)

# create sinogram with ccpi-astra
sin_ccpi = Aop.direct(X)  

# create sinogram with astra
sinogram_id, sin_astra = astra.create_sino3d_gpu(X.as_array(), proj_geom3D, vol_geom3D)
astra.data3d.delete(sinogram_id)

# assert sinograms 
np.testing.assert_array_equal(sin_ccpi.as_array(), sin_astra)

#%% check fbp

filter_type = 'ram-lak'

# fbp with ccpi-astra
fbp_ccpi = Aop.FBP(sin_ccpi, filter_type)
   
# fbp with astra
proj_id = astra.create_projector('cuda', proj_geom2D, vol_geom2D) 
rec = np.zeros(X.shape)

for i in range(64):

    sinogram_id = astra.data2d.create('-sino', proj_geom2D, sin_ccpi.as_array()[i])
    rec_id = astra.data2d.create('-vol', vol_geom2D)

    cfg = astra.astra_dict('FBP_CUDA')
    
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['FilterType'] = filter_type
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec[i] = astra.data2d.get(rec_id) * voxels
    
#astra.algorithm.delete(alg_id)
#astra.data2d.delete(rec_id)
#astra.data2d.delete(sinogram_id)
#astra.projector.delete(proj_id)    

slice_ind = int(N_size/2)    

plt.figure(figsize = (10,30)) 
plt.subplot(131)
plt.imshow(fbp_ccpi.as_array()[slice_ind,:,:])
plt.colorbar()
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(fbp_ccpi.as_array()[:,slice_ind,:])
plt.colorbar()
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(fbp_ccpi.as_array()[:,:,slice_ind])
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

# assert fbp astra and ccpi-astra 
np.testing.assert_array_equal(fbp_ccpi.as_array(), rec)

#%% Now test FDK

#mag = (ag.dist_source_center + ag.dist_center_detector)/ag.dist_source_center

distance_source_origin = 300  # [mm]
distance_origin_detector = 100  # [mm]
detector_pixel_size = 1.05  # [mm]
detector_rows = 200  # Vertical size of detector [pixels].
detector_cols = 200  # Horizontal size of detector [pixels].

voxels = 1

mag = (distance_source_origin + distance_origin_detector) / distance_source_origin

voxel_real = detector_pixel_size / mag

ig_cone = ImageGeometry(voxel_num_x=N_size, voxel_num_y=N_size, voxel_num_z=N_size,
                        voxel_size_x = voxels,
                        voxel_size_y = voxels,
                        voxel_size_z = voxels)

ag_cone = AcquisitionGeometry(geom_type = 'cone', dimension = '3D', 
                         angles = angles_rad, pixel_num_h=Horiz_det, 
                         pixel_num_v=Vert_det, 
                         pixel_size_h = voxels * mag,
                         pixel_size_v = voxels * mag,
                         dist_center_detector = distance_origin_detector/voxel_real, 
                         dist_source_center= distance_source_origin/voxel_real, 
                         dimension_labels=['vertical','angle','horizontal'])



Aop_cone = AstraProjector3DSimple(ig_cone, ag_cone)

sin_cone = Aop_cone.direct(X)


fdk = Aop_cone.FBP(sin_cone, 'ram-lak')
plt.imshow(fdk.as_array()[32])
plt.colorbar()
plt.show()


vol_geom_cone, proj_geom_cone = convert_geometry_to_astra(ig_cone, ag_cone)
          
rec_id = astra.data3d.create('-vol', vol_geom_cone)
sinogram_id = astra.data3d.create('-sino', proj_geom_cone, sin_cone.as_array())

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
    
rec = astra.data3d.get(rec_id) #/ (voxels**3)
astra.data3d.delete(rec_id)
astra.data3d.delete(sinogram_id)
            
astra.algorithm.delete(alg_id)

plt.imshow(rec[32])
plt.colorbar()
plt.show()

# assert fdk astra and ccpi-astra 
np.testing.assert_array_equal(fdk.as_array(), rec)
