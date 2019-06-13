
# This demo shows how to load a Nikon XTek micro-CT data set and reconstruct
# a volume of 200 slices using the CGLS method. The SophiaBeads dataset with 64 
# projections is used as test data and can be obtained from here:
# https://zenodo.org/record/16474
# The filename with full path to the .xtekct file should be given as string 
# input to NikonDataReader to  load in the data.

# Do all imports
import numpy as np
import matplotlib.pyplot as plt
from ccpi.io import NikonDataReader
from ccpi.framework import ImageGeometry, ImageData
from ccpi.astra.operators import AstraProjector3DSimple
from ccpi.optimisation.algorithms import CGLS

# Set up reader object and read in 200 central slices of the data
datareader= NikonDataReader(xtek_file="REPLACE_THIS_BY_PATH_TO_DATASET/SophiaBeads_64_averaged.xtekct",
                            roi=[(901,1101),(0,2000)])
data = datareader.load_projections()

# Scale and negative-log transform
data.fill(-np.log(data.as_array()/60000.0))

# Apply centering correction by zero padding, amount found manually
cor_pad = 30
data_pad = np.zeros((data.shape[0],data.shape[1],data.shape[2]+cor_pad))
data_pad[:,:,cor_pad:] = data.array
data.geometry.pixel_num_h = data.geometry.pixel_num_h + cor_pad
data.array = data_pad

# Choose the number of voxels to reconstruct onto as number of detector pixels
N = data.geometry.pixel_num_h

# Geometric magnification
mag = (np.abs(data.geometry.dist_center_detector) + \
      np.abs(data.geometry.dist_source_center)) / \
      np.abs(data.geometry.dist_source_center)

# Voxel size is detector pixel size divided by mag
voxel_size_h = data.geometry.pixel_size_h / mag

# Construct the appropriate ImageGeometry
ig = ImageGeometry(voxel_num_x=N,
                   voxel_num_y=N,
                   voxel_num_z=data.geometry.pixel_num_v,
                   voxel_size_x=voxel_size_h, 
                   voxel_size_y=voxel_size_h,
                   voxel_size_z=voxel_size_h)

# Permute array and convert angles to radions for ASTRA; delete old data.
data2 = data.subset(dimensions=['vertical','angle','horizontal'])
data2.geometry = data.geometry
data2.geometry.angles = -data2.geometry.angles/180*np.pi
del data

# Extract the Acquisition geometry for setting up projector.
ag = data2.geometry

# Set up the Projector (AcquisitionModel) using ASTRA on GPU
Aop = AstraProjector3DSimple(ig, ag)

# Do and show simple backprojection
z = Aop.adjoint(data2)
plt.figure()
plt.imshow(z.subset(horizontal_x=1000).as_array())
plt.show()
plt.figure()
plt.imshow(z.subset(horizontal_y=1000).as_array())
plt.show()
plt.figure()
plt.imshow(z.subset(vertical=100).array)
plt.show()

# Set initial guess for CGLS reconstruction
x_init = ImageData(geometry=ig)

# Set tolerance and number of iterations for reconstruction algorithms.
opt = {'tol': 1e-4, 'iter': 30}

# Run a CGLS reconstruction can be done:
CGLS_alg = CGLS()
CGLS_alg.set_up(x_init, Aop, data2)
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])

x_CGLS = CGLS_alg.get_output()

# Display ortho slices of reconstruction
plt.figure()
plt.imshow(x_CGLS.subset(horizontal_x=1000).as_array())
plt.show()
plt.figure()
plt.imshow(x_CGLS.subset(horizontal_y=1000).as_array())
plt.show()
plt.figure()
plt.imshow(x_CGLS.subset(vertical=100).as_array())
plt.show()

plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()