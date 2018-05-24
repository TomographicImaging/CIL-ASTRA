
# This demo shows how to load a Nikon XTek micro-CT data set and reconstruct
# the central slice using the CGLS method. The SophiaBeads dataset with 256 
# projections is used as test data and can be obtained from here:
# https://zenodo.org/record/16474
# The filename with full path to the .xtekct file should be given as string 
# input to XTEKReader to  load in the data.

# Do all imports
from ccpi.io.reader import XTEKReader
import numpy as np
import matplotlib.pyplot as plt
from ccpi.framework import ImageGeometry, AcquisitionData, ImageData
from ccpi.astra.ops import AstraProjector3DSimple
from ccpi.optimisation.algs import CGLS

import numpy

# Set up reader object and read the data
datareader = XTEKReader("REPLACE_THIS_BY_PATH_TO_DATASET/SophiaBeads_256_averaged.xtekct")
data = datareader.get_acquisition_data()

# Crop data and fix dimension labels
data = AcquisitionData(data.array[:,:,901:1101],
                            geometry=data.geometry,
                            dimension_labels=['angle','horizontal','vertical'])
data.geometry.pixel_num_v = 200

# Scale and negative-log transform
data.array = -np.log(data.as_array()/60000.0)

# Apply centering correction by zero padding, amount found manually
cor_pad = 30
data_pad = np.zeros((data.shape[0],data.shape[1]+cor_pad,data.shape[2]))
data_pad[:,cor_pad:,:] = data.array
data.geometry.pixel_num_h = data.geometry.pixel_num_h + cor_pad
data.array = data_pad

# Simple beam hardening correction as done in SophiaBeads coda
#data = data**2

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
data2.geometry.angles = -data2.geometry.angles/180*numpy.pi
del data

# Extract the Acquisition geometry for setting up projector.
ag = data2.geometry

# Set up the Projector (AcquisitionModel) using ASTRA on GPU
Aop = AstraProjector3DSimple(ig, ag)

# So and show simple backprojection
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

# Run 50-iteration CGLS reconstruction
opt = {'tol': 1e-4, 'iter': 20}
x, it, timing, criter = CGLS(x_init,Aop,data2,opt=opt)

# Display ortho slices of reconstruction
plt.figure()
plt.imshow(x.subset(horizontal_x=1000).as_array())
plt.show()
plt.figure()
plt.imshow(x.subset(horizontal_y=1000).as_array())
plt.show()
plt.figure()
plt.imshow(x.subset(vertical=100).as_array())
plt.show()
