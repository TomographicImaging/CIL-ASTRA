
# This demo demonstrates a simple multichannel reconstruction case. A 
# synthetic 3-channel phantom image is set up, data is simulated and the FISTA 
# algorithm is used to compute least squares and least squares with 1-norm 
# regularisation reconstructions.

# Do all imports
from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algs import FISTA
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.astra.ops import AstraProjectorMC

import numpy
import matplotlib.pyplot as plt

# Choose either a parallel-beam (1=parallel2D) or fan-beam (2=cone2D) test case
test_case = 1

# Set up phantom NxN pixels and 3 channels. Set up the ImageGeometry and fill 
# some test data in each of the channels. Display each channel as image.
N = 128
numchannels = 3

ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N,channels=numchannels)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[0 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 1.0
x[0 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 2.0

x[1 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 0.7
x[1 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 1.2

x[2 , round(N/4):round(3*N/4) , round(N/4):round(3*N/4)  ] = 1.5
x[2 , round(N/8):round(7*N/8) , round(3*N/8):round(5*N/8)] = 2.2

f, axarr = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarr[k].imshow(x[k],vmin=0,vmax=2.5)
plt.show()

# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).
angles_num = 20
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

if test_case==1:
    angles = numpy.linspace(0,numpy.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,
                             det_w,
                             channels=numchannels)
elif test_case==2:
    angles = numpy.linspace(0,2*numpy.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                            '2D',
                            angles,
                            det_num,
                            det_w,
                            dist_source_center=SourceOrig, 
                            dist_center_detector=OrigDetec,
                            channels=numchannels)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorMC(ig, ag, 'gpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z. Applies 
# channel by channel
b = Aop.direct(Phantom)

fb, axarrb = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrb[k].imshow(b.as_array()[k],vmin=0,vmax=250)
plt.show()

z = Aop.adjoint(b)

fo, axarro = plt.subplots(1,numchannels)
for k in range(3):
    axarro[k].imshow(z.as_array()[k],vmin=0,vmax=3500)
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 200}

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5. Note it is least squares over all channels:
f = Norm2sq(Aop,b,c=0.5)

# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt)

# Display reconstruction and criteration
ff0, axarrf0 = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrf0[k].imshow(x_fista0.as_array()[k],vmin=0,vmax=2.5)
plt.show()

plt.semilogy(criter0)
plt.title('Criterion vs iterations, least squares')
plt.show()

# FISTA can also solve regularised forms by specifying a second function object
# such as 1-norm regularisation with choice of regularisation parameter lam. 
# Again the regulariser is over all channels:
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

# Display reconstruction and criteration
ff1, axarrf1 = plt.subplots(1,numchannels)
for k in numpy.arange(3):
    axarrf1[k].imshow(x_fista1.as_array()[k],vmin=0,vmax=2.5)
plt.show()

plt.semilogy(criter1)
plt.title('Criterion vs iterations, least squares plus 1-norm regu')
plt.show()
