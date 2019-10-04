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
Aop2D = AstraProjectorSimple(ig2D, ag2D, 'gpu')
# Add noise to the sinogram data
X = ImageData(phantom_tm, geometry = ig)
sin = Aop.direct(X)


tmp = Aop.domain_geometry().allocate()
#
for i in range(sin.shape[0]):
    print(i)
    tmp1 = Aop2D.FBP(sin.subset(vertical=i), 'ram-lak')
    np.copyto(tmp.array[i], tmp1.array)
    

#%%    
slice_ind = int(N_size/2)    
plt.figure(figsize = (10,30)) 
plt.subplot(131)
plt.imshow(tmp.as_array()[slice_ind,:,:])
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(tmp.as_array()[:,slice_ind,:])
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(tmp.as_array()[:,:,slice_ind])
plt.title('3D Phantom, sagittal view')
plt.show()