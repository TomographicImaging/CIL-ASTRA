
# This demo illustrates how ASTRA 2D projectors can be used with
# the modular optimisation framework. The demo sets up a 2D test case and 
# demonstrates reconstruction using CGLS, as well as FISTA for least squares 
# and 1-norm regularisation and FBPD for 1-norm and TV regularisation.

# First make all imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D
from ccpi.astra.operators import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt

# Choose either a parallel-beam (1=parallel2D) or fan-beam (2=cone2D) test case
test_case = 1

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 128
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
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
plt.show()

plt.imshow(z.array)
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set:
x_init = ImageData(np.zeros(x.shape),geometry=ig)
opt = {'tol': 1e-4, 'iter': 1000}

# First a CGLS reconstruction can be done:
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, b, opt)

plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()

# CGLS solves the simple least-squares problem. The same problem can be solved 
# by FISTA by setting up explicitly a least squares function object and using 
# no regularisation:

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2sq(Aop,b,c=0.5)

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

# The least squares plus 1-norm regularisation problem can also be solved by 
# other algorithms such as the Forward Backward Primal Dual algorithm. This
# algorithm minimises the sum of three functions and the least squares and 
# 1-norm functions should be given as the second and third function inputs. 
# In this test case, this algorithm requires more iterations to converge, so
# new options are specified.
opt_FBPD = {'tol': 1e-4, 'iter': 20000}
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt_FBPD)

plt.imshow(x_fbpd1.array)
plt.title('FBPD for least squares plus 1-norm regularisation')
plt.show()

plt.semilogy(criter_fbpd1)
plt.title('FBPD for least squares plus 1-norm regularisation criterion')
plt.show()

# The FBPD algorithm can also be used conveniently for TV regularisation:

# Specify TV function object
lamtv = 10
gtv = TV2D(lamtv)

x_fbpdtv,it_fbpdtv,timing_fbpdtv,criter_fbpdtv=FBPD(x_init,None,f,gtv,opt_FBPD)

plt.imshow(x_fbpdtv.array)
plt.show()

plt.semilogy(criter_fbpdtv)
plt.show()


# Compare all reconstruction and criteria
clims = (0,1)
cols = 3
rows = 2
current = 1

fig = plt.figure()
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))
imgplot = plt.imshow(Phantom.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS')
imgplot = plt.imshow(x_fista0.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_fista1.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD LS+1')
imgplot = plt.imshow(x_fbpd1.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD TV')
imgplot = plt.imshow(x_fbpdtv.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

fig = plt.figure()
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
imgplot = plt.loglog(criter0 , label='FISTA LS')
imgplot = plt.loglog(criter1 , label='FISTA LS+1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD LS+1')
imgplot = plt.loglog(criter_fbpdtv, label='FBPD TV')
b.legend(loc='lower left')
plt.show()
