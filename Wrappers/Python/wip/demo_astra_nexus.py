
# This script demonstrates how to load a parallel beam data set in Nexus 
# format, apply dark and flat field correction and reconstruct using the
# modular optimisation framework and the ASTRA Tomography toolbox.
# 
# The data set is available from
# https://github.com/DiamondLightSource/Savu/blob/master/test_data/data/24737_fd.nxs
# and should be downloaded to a local directory to be specified below.

# All own imports
from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algorithms import CGLS, FISTA
from ccpi.optimisation.functions import Norm2Sq, L1Norm
from ccpi.processors import Normalizer, CenterOfRotationFinder 
from ccpi.io.reader import NexusReader
from ccpi.astra.operators import AstraProjector3DSimple

# All external imports
import numpy
import matplotlib.pyplot as plt
import os

# Define utility function to average over flat and dark images.
def avg_img(image):
    shape = list(numpy.shape(image))
    l = shape.pop(0)
    avg = numpy.zeros(shape)
    for i in range(l):
        avg += image[i] / l
    return avg
    
# Set up a reader object pointing to the Nexus data set. Revise path as needed.
reader = NexusReader(os.path.join(".." ,".." ,".." , "..", "CCPi-ReconstructionFramework","data" , "24737_fd.nxs" ))

# Read and print the dimensions of the raw projections
dims = reader.get_projection_dimensions()
print (dims)

# Load and average all flat and dark images in preparation for normalising data.
flat = avg_img(reader.load_flat())
dark = avg_img(reader.load_dark())

# Set up normaliser object for normalising data by flat and dark images.
norm = Normalizer(flat_field=flat, dark_field=dark)

# Load the raw projections and pass as input to the normaliser and apply 
# normlisation.
norm.set_input(reader.get_acquisition_data())
data = norm.get_output()
data.array = -numpy.log(data.as_array())

# Set up CenterOfRotationFinder object to center data.
# Set the output of the normaliser as the input and execute to determine center.
cor = CenterOfRotationFinder()
cor.set_input(data)
center_of_rotation = cor.get_output()

# From computed center, determine amount of zero-padding to apply, apply
# and update geometry to wider detector.
cor_pad = int(2*(center_of_rotation - data.shape[2]/2))
data_pad = numpy.zeros((data.shape[0],data.shape[1],data.shape[2]+cor_pad))
data_pad[:,:,:-cor_pad] = data.as_array()
data.geometry.pixel_num_h = data.geometry.pixel_num_h + cor_pad
data.array = data_pad

# Permute array and convert angles to radions for ASTRA
padded_data = data.subset(dimensions=['vertical','angle','horizontal'])
padded_data.geometry = data.geometry
padded_data.geometry.angles = padded_data.geometry.angles/180*numpy.pi

# Create Acquisition and Image Geometries for setting up projector.
ag = padded_data.geometry
ig = ImageGeometry(voxel_num_x=ag.pixel_num_h,
                   voxel_num_y=ag.pixel_num_h, 
                   voxel_num_z=ag.pixel_num_v)

# Define the projector object
print ("Define projector")
Cop = AstraProjector3DSimple(ig, ag)

# Test backprojection and projection
z1 = Cop.adjoint(padded_data)
z2 = Cop.direct(z1)

plt.imshow(z1.subset(horizontal_x=68).array)
plt.show()

# Set initial guess for reconstruction algorithms.
print ("Initial guess")
x_init = ImageData(geometry=ig)

# Set tolerance and number of iterations for reconstruction algorithms.
opt = {'tol': 1e-4, 'iter': 100}

# First a CGLS reconstruction can be done:
CGLS_alg = CGLS()
CGLS_alg.set_up(x_init, Cop, padded_data)
CGLS_alg.max_iteration = 2000
CGLS_alg.run(opt['iter'])

x_CGLS = CGLS_alg.get_output()

# Fix color and slices for display
v1 = -0.01
v2 = 0.13
hx=80
hy=80
v=68

# Display ortho slices of reconstruction
# Display all reconstructions and decay of objective function
cols = 3
rows = 1
fig = plt.figure()

current = 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_x')
imgplot = plt.imshow(x_CGLS.subset(horizontal_x=hx).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_y')
imgplot = plt.imshow(x_CGLS.subset(horizontal_y=hy).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('vertical')
imgplot = plt.imshow(x_CGLS.subset(vertical=v).as_array(),vmin=v1,vmax=v2)
plt.colorbar()

plt.suptitle('CGLS reconstruction slices')
plt.show()

plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()

# Create least squares object instance with projector and data.
print ("Create least squares object instance with projector and data.")
f = Norm2Sq(Cop,padded_data,c=0.5)


# Run FISTA for least squares without constraints
FISTA_alg = FISTA()
FISTA_alg.set_up(x_init=x_init, f=f, opt=opt)
FISTA_alg.max_iteration = 2000
FISTA_alg.run(opt['iter'])
x_FISTA = FISTA_alg.get_output()

# Display ortho slices of reconstruction
# Display all reconstructions and decay of objective function

fig = plt.figure()

current = 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_x')
imgplot = plt.imshow(x_FISTA.subset(horizontal_x=hx).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_y')
imgplot = plt.imshow(x_FISTA.subset(horizontal_y=hy).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('vertical')
imgplot = plt.imshow(x_FISTA.subset(vertical=v).as_array(),vmin=v1,vmax=v2)
plt.colorbar()

plt.suptitle('FISTA Least squares reconstruction slices')
plt.show()

plt.figure()
plt.semilogy(FISTA_alg.objective)
plt.title('FISTA Least squares criterion')
plt.show()

# Create 1-norm function object
lam = 30.0
g0 = lam * L1Norm()

# Run FISTA for least squares plus 1-norm function.
FISTA_alg1 = FISTA()
FISTA_alg1.set_up(x_init=x_init, f=f, g=g0, opt=opt)
FISTA_alg1.max_iteration = 2000
FISTA_alg1.run(opt['iter'])
x_FISTA1 = FISTA_alg1.get_output()

# Display all reconstructions and decay of objective function
fig = plt.figure()

current = 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_x')
imgplot = plt.imshow(x_FISTA1.subset(horizontal_x=hx).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('horizontal_y')
imgplot = plt.imshow(x_FISTA1.subset(horizontal_y=hy).as_array(),vmin=v1,vmax=v2)

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('vertical')
imgplot = plt.imshow(x_FISTA1.subset(vertical=v).as_array(),vmin=v1,vmax=v2)
plt.colorbar()

plt.suptitle('FISTA LS+1 reconstruction slices')
plt.show()


plt.figure()
plt.semilogy(FISTA_alg1.objective)
plt.title('FISTA LS+1 criterion')
plt.show()


fig = plt.figure()
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(CGLS_alg.objective , label='CGLS')
imgplot = plt.loglog(FISTA_alg.objective , label='FISTA LS')
imgplot = plt.loglog(FISTA_alg1.objective, label='FISTA LS+1')
b.legend(loc='right')
plt.show()
