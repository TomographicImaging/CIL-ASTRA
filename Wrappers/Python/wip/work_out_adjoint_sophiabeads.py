
# This demo illustrates how ASTRA 2D projectors can be used with
# the modular optimisation framework. The demo sets up a 2D test case and 
# demonstrates reconstruction using CGLS, as well as FISTA for least squares 
# and 1-norm regularisation and FBPD for 1-norm and TV regularisation.

# First make all imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D
from ccpi.astra.operators import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt

# Choose either a parallel-beam (1=parallel2D) or fan-beam (2=cone2D) test case
test_case = 2

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 2000
x1 = -16.015690364093633
x2 =  16.015690364093633
dx = (x2-x1)/N
ig = ImageGeometry(voxel_num_x=N,
                   voxel_num_y=N,
                   voxel_size_x=dx,
                   voxel_size_y=dx)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[:,round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.colorbar()
plt.show()

# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).
angles_num = 4
det_num = 2500

SourceOrig = 80.6392412185669
OrigDetec = 926.3637587814331

geo_mag = (SourceOrig+OrigDetec)/SourceOrig

det_w = geo_mag*dx*1

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,det_w)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             det_num,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorSimple(ig, ag, 'gpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

plt.imshow(b.array)
plt.title('Simulated data')
plt.colorbar()
plt.show()

plt.imshow(z.array)
plt.title('Backprojected data')
plt.colorbar()
plt.show()


One = ImageData(geometry=ig)
xOne = One.as_array()
xOne[:,:] = 1.0

OneD = AcquisitionData(geometry=ag)
y1 = OneD.as_array()
y1[:,:] = 1.0

s1 = (OneD*(Aop.direct(One))).sum()
s2 = (One*(Aop.adjoint(OneD))).sum()
print(s1)
print(s2)
print(s2/s1)

print((b*b).sum())
print((z*Phantom).sum())
print((z*Phantom).sum() / (b*b).sum())

#print(N/det_num)
#print(0.5*det_w/dx)