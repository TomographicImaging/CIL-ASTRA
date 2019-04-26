
# This script demonstrates how to load a parallel beam data set in Nexus 
# format, apply dark and flat field correction and reconstruct using the
# modular optimisation framework and the ASTRA Tomography toolbox.
# 
# The data set is available from
# https://github.com/DiamondLightSource/Savu/blob/master/test_data/data/24737_fd.nxs
# and should be downloaded to a local directory to be specified below.

# All own imports
from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.processors import Normalizer, CenterOfRotationFinder 
from ccpi.plugins.processors import AcquisitionDataPadder
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

# Load the raw projections and pass as input to the normaliser.
norm.set_input(reader.get_acquisition_data())

# Set up CenterOfRotationFinder object to center data.
cor = CenterOfRotationFinder()

# Set the output of the normaliser as the input and execute to determine center.
cor.set_input(norm.get_output())
center_of_rotation = cor.get_output()

# Set up AcquisitionDataPadder to pad data for centering using the computed 
# center, set the output of the normaliser as input and execute to produce
# padded/centered data.
padder = AcquisitionDataPadder(center_of_rotation=center_of_rotation)
padder.set_input(norm.get_output())
padded_data = padder.get_output()

# Permute array and convert angles to radions for ASTRA
padded_data2 = padded_data.subset(dimensions=['vertical','angle','horizontal'])
padded_data2.geometry = padded_data.geometry
padded_data2.geometry.angles = padded_data2.geometry.angles/180*numpy.pi

# Create Acquisition and Image Geometries for setting up projector.
ag = padded_data2.geometry
ig = ImageGeometry(voxel_num_x=ag.pixel_num_h,
                   voxel_num_y=ag.pixel_num_h, 
                   voxel_num_z=ag.pixel_num_v)

# Define the projector object
print ("Define projector")
Cop = AstraProjector3DSimple(ig, ag)

# Test backprojection and projection
z1 = Cop.adjoint(padded_data2)
z2 = Cop.direct(z1)

plt.imshow(z1.subset(vertical=68).array)
plt.show()

# Set initial guess
print ("Initial guess")
x_init = ImageData(geometry=ig)

# Create least squares object instance with projector and data.
print ("Create least squares object instance with projector and data.")
f = Norm2sq(Cop,padded_data2,c=0.5)
        
# Run FISTA reconstruction for least squares without regularization
print ("Run FISTA for least squares")
opt = {'tol': 1e-4, 'iter': 100}
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt=opt)

plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA LS')
plt.colorbar()
plt.show()

# Set up 1-norm function for FISTA least squares plus 1-norm regularisation
print ("Run FISTA for least squares plus 1-norm regularisation")
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0,opt=opt)

plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA LS+1')
plt.colorbar()
plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
print ("Run FBPD for least squares plus 1-norm regularisation")
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.subset(horizontal_x=80).array)
plt.title('FBPD LS+1')
plt.colorbar()
plt.show()

# Run CGLS, which should agree with the FISTA least squares
print ("Run CGLS for least squares")
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Cop, padded_data2, opt=opt)
plt.imshow(x_CGLS.subset(horizontal_x=80).array)
plt.title('CGLS')
plt.colorbar()
plt.show()

# Display all reconstructions and decay of objective function
cols = 4
rows = 1
current = 1
fig = plt.figure()

current = current 
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS')
imgplot = plt.imshow(x_fista0.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_fista1.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD LS+1')
imgplot = plt.imshow(x_fbpd1.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.subset(horizontal_x=80).as_array())

plt.show()

fig = plt.figure()
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter0 , label='FISTA LS')
imgplot = plt.loglog(criter1 , label='FISTA LS+1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD LS+1')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
b.legend(loc='right')
plt.show()
