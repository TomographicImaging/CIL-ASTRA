
# This demo shows how to load a Nikon XTek micro-CT data set and reconstruct
# the central slice using the CGLS and FISTA methods. The SophiaBeads dataset 
# with 64 projections is used as test data and can be obtained from here:
# https://zenodo.org/record/16474
# The filename with full path to the .xtekct file should be given as string 
# input to NikonDataReader to  load in the data.

# Do all imports
import numpy as np
import matplotlib.pyplot as plt
from ccpi.io import NikonDataReader
from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData, ImageData
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.optimisation.algorithms import FISTA, CGLS
from ccpi.optimisation.functions import Norm2Sq, L1Norm

# Set up reader object and read the data
datareader_new = NikonDataReader(xtek_file="REPLACE_THIS_BY_PATH_TO_DATASET/SophiaBeads_64_averaged.xtekct")
data = datareader_new.load_projections()

# Extract central slice, scale and negative-log transform
sino = -np.log(data.as_array()[:,1000,:]/60000.0)

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
x_init = ImageData(geometry=ig2d)

# Set tolerance and number of iterations for reconstruction algorithms.
opt = {'tol': 1e-4, 'iter': 50}

# First a CGLS reconstruction can be done:
CGLS_alg = CGLS()
CGLS_alg.set_up(x_init, Aop, data2d)
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])

x_CGLS = CGLS_alg.get_output()

plt.figure()
plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()

# CGLS solves the simple least-squares problem. The same problem can be solved 
# by FISTA by setting up explicitly a least squares function object and using 
# no regularisation:

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2Sq(Aop,data2d)

# Run FISTA for least squares without constraints
FISTA_alg = FISTA()
FISTA_alg.set_up(x_init=x_init, f=f, opt=opt)
FISTA_alg.max_iteration = 2000
FISTA_alg.run(opt['iter'])
x_FISTA = FISTA_alg.get_output()

plt.figure()
plt.imshow(x_FISTA.array)
plt.title('FISTA Least squares reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg.objective)
plt.title('FISTA Least squares criterion')
plt.show()

# Create 1-norm function object
lam = 1.0
g0 = lam * L1Norm()

# Run FISTA for least squares plus 1-norm function.
FISTA_alg1 = FISTA()
FISTA_alg1.set_up(x_init=x_init, f=f, g=g0, opt=opt)
FISTA_alg1.max_iteration = 2000
FISTA_alg1.run(opt['iter'])
x_FISTA1 = FISTA_alg1.get_output()

plt.figure()
plt.imshow(x_FISTA1.array)
plt.title('FISTA LS+L1Norm reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg1.objective)
plt.title('FISTA LS+L1norm criterion')
plt.show()