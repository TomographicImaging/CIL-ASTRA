
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
from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData, ImageData
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.algs import CGLS

# Set up reader object and read the data
datareader = XTEKReader("REPLACE_THIS_BY_PATH_TO_DATASET/SophiaBeads_256_averaged.xtekct")
data = datareader.get_acquisition_data()

# Extract central slice, scale and negative-log transform
sino = -np.log(data.as_array()[:,:,1000]/60000.0)

# Apply centering correction by zero padding, amount found manually
cor_pad = 30
sino_pad = np.zeros((sino.shape[0],sino.shape[1]+cor_pad))
sino_pad[:,cor_pad:] = sino

# Extract AcquisitionGeometry for central slice for 2D fanbeam reconstruction
ag2d = AcquisitionGeometry('cone',
                          '2D',
                          angles=-np.pi/180*data.geometry.angles,
                          pixel_num_h=data.geometry.pixel_num_h + cor_pad,
                          pixel_size_h=data.geometry.pixel_size_h,
                          dist_source_center=-data.geometry.dist_source_center, 
                          dist_center_detector=data.geometry.dist_center_detector)

# Set up AcquisitionData object for central slice 2D fanbeam
data2d = AcquisitionData(sino_pad,geometry=ag2d)

# Choose the number of voxels to reconstruct onto as number of detector pixels
N = data.geometry.pixel_num_h

# Geometric magnification
mag = (np.abs(data.geometry.dist_center_detector) + \
      np.abs(data.geometry.dist_source_center)) / \
      np.abs(data.geometry.dist_source_center)

# Voxel size is detector pixel size divided by mag
voxel_size_h = data.geometry.pixel_size_h / mag

# Construct the appropriate ImageGeometry
ig2d = ImageGeometry(voxel_num_x=N,
                   voxel_num_y=N,
                   voxel_size_x=voxel_size_h, 
                   voxel_size_y=voxel_size_h)

# Set up the Projector (AcquisitionModel) using ASTRA on GPU
Aop = AstraProjectorSimple(ig2d, ag2d,"gpu")

# Set initial guess for CGLS reconstruction
x_init = ImageData(np.zeros((N,N)),geometry=ig2d)

# Run 50-iteration CGLS reconstruction
opt = {'tol': 1e-4, 'iter': 50}
x, it, timing, criter = CGLS(x_init,Aop,data2d,opt=opt)

# Display reconstruction
plt.figure()
plt.imshow(x.as_array())
plt.colorbar()
plt.show()




##############################################

from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D

opt = {'tol': 1e-4, 'iter': 100}

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2sq(Aop,data2d,c=0.5)

# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None,opt)

plt.imshow(x_fista0.array)
plt.title('FISTA Least squares')
plt.show()

plt.semilogy(criter0)
plt.title('FISTA Least squares criterion')
plt.show()

# FISTA can also solve regularised forms by specifying a second function object
# such as 1-norm regularisation with choice of regularisation parameter lam:

# Create 1-norm function object
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

plt.imshow(x_fista1.array)
plt.title('FISTA Least squares plus 1-norm regularisation')
plt.show()

plt.semilogy(criter1)
plt.title('FISTA Least squares plus 1-norm regularisation criterion')
plt.show()

# Specify TV function object
lamtv = 0.1
gtv = TV2D(lamtv)

optTV = {'tol': 1e-4, 'iter': 10000}

x_fbpdtv,it_fbpdtv,timing_fbpdtv,criter_fbpdtv=FBPD(x_init,None,f,gtv,optTV)

plt.imshow(x_fbpdtv.array)
plt.show()

plt.semilogy(criter_fbpdtv)
plt.show()