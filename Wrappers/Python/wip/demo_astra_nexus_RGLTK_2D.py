
# This script demonstrates how to load a parallel beam data set in Nexus 
# format, apply dark and flat field correction and reconstruct using the
# modular optimisation framework and the ASTRA Tomography toolbox.
# 
# The data set is available from
# https://github.com/DiamondLightSource/Savu/blob/master/test_data/data/24737_fd.nxs
# and should be downloaded to a local directory to be specified below.

# All own imports
from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, AcquisitionGeometry, DataContainer
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.plugins.processors import AcquisitionDataPadder
#from ccpi.plugins.regularisers import ROF_TV, FGP_TV, SB_TV

from ccpi.io.reader import NexusReader
from ccpi.astra.ops import AstraProjectorSimple

# All external imports
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from astropy.io import fits

# Define utility function to average over flat and dark images.
def avg_img(image):
    shape = list(np.shape(image))
    l = shape.pop(0)
    avg = np.zeros(shape)
    for i in range(l):
        avg += image[i] / l
    return avg
    
##############################################################################
#%%
# Read data once
# Set up a reader object pointing to the Nexus data set. Revise path as needed.
filename = '74143'
reader = NexusReader('{!s}{!s}.nxs'.format("/home/algol/Documents/DATA/",filename))
#reader = NexusReader("/home/algol/Documents/DATA/24737_fd.nxs")
dims = reader.get_sinogram_dimensions()
#print (reader.get_sinogram_dimensions())
a = reader.load()
# plt.imshow(a[:,100,:])

# Set paths to darks and flats here
f_d = h5py.File("/home/algol/Documents/DATA/74240.nxs",'r')
darks = list(f_d['/entry1/instrument/pco1_hw_hdf_nochunking/data'])
f_f = h5py.File("/home/algol/Documents/DATA/74243.nxs",'r')
flats = list(f_f['/entry1/instrument/pco1_hw_hdf_nochunking/data'])

shapeFl = np.shape(darks)
av_darks = np.zeros(np.shape(darks[0]),'float32')
av_flats = np.zeros(np.shape(darks[0]),'float32')
for jj in range(shapeFl[0]):
     av_darks += np.float32(darks[jj])
     av_flats += np.float32(flats[jj])

av_darks = av_darks/np.float32(shapeFl[0])
av_flats =av_flats/np.float32(shapeFl[0])

#flat = np.load('flats.npy')
#dark = np.load('darks.npy')
#sinoExtr = np.load('sino.npy')
#%%
##############################################################################
N = 2000  # reconstruction size [N x N]
SliceFirst = 500 # First (bottom) vertical slice of the dataset
SliceLast = 502 # Last (top) vertical slice of the dataset
TotalSlices  = SliceLast-SliceFirst

counter = SliceFirst
for i in range(TotalSlices):
    sinoExtr = a[:,counter,:] # selecting particular slice
    angles_num = dims[1]
    det_num = dims[2]
    angles = np.pi/180*reader.get_projection_angles()
    det_w = 1
    
    flat1D = av_flats[counter,:]
    dark1D = av_darks[counter,:]
    
    # normalising and taking neg log of data
    res = flat1D - dark1D
    indx = res > 0
    sino_norm = np.zeros(np.shape(sinoExtr), 'float32')
    sino_norm[0:angles_num,indx] = (sinoExtr[0:angles_num,indx] - dark1D[indx])/(res[indx])
    indx0 = res==0
    sino_norm[0:angles_num,indx0] = 1
    sino_norm[sino_norm <= 0] = 1
    sino_norm_log = -np.log(sino_norm)*1000

    #plt.figure()
    #plt.imshow((sino_norm_log), vmin=0, vmax=1000)
    #plt.title('Log corrected normalised sinogram')
    #plt.show()
##############################################################################
    # Apply centering correction by zero padding, amount found manually
    # cor_pad = 110 # this value needs to be corrected (for 240 data)
    cor_pad = 150 # this value needs to be corrected 
    sino_pad = np.zeros((sino_norm_log.shape[0],sino_norm_log.shape[1]+cor_pad))
    sino_pad[:,cor_pad:] = sino_norm_log
    #sino_pad[:,: sino_norm_log.shape[1]] = sino_norm_log

    det_num = sino_pad.shape[1]

    ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,det_w)
    Aop = AstraProjectorSimple(ig, ag, 'gpu')
##############################################################################
# Reconstruction using Filtered BackProjection (FBP) Method (2D)
    """
    FBP = Aop.FBP(DataContainer(sino_pad), filter_type = 'Shepp-Logan')
    plt.figure()
    plt.imshow(FBP.array, vmin=0, vmax=1)
    plt.title('FBP reconstructed image')
    plt.show()
    """
##############################################################################
    # Reconstruct iteratively using FISTA - TV algorithm
    # Set initial guess
    print ("Initial guess")
    x_init = ImageData(geometry=ig)

    # Create least squares object instance with projector and data.
    print ("Create least squares object instance with projector and data.")
    f = Norm2sq(Aop,DataContainer(sino_pad),c=0.5)
    """
    lamtv = 2000
    opt = {'tol': 1e-4, 'iter': 85}
    g_fgp = FGP_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-6,
                 methodTV=0,
                 nonnegativity=0,
                 printing=0,
                 device='gpu')

    x_fista_fgp, it1, timing1, criter_fgp = FISTA(x_init, f, g_fgp, opt)

    plt.figure()
    plt.subplot(121)
    plt.imshow(x_fista_fgp.array, vmin=0, vmax=1)
    plt.title('FISTA FGP TV')
    plt.subplot(122)
    plt.semilogy(criter_fgp)
    plt.show()
    """
    print ("Run CGLS for least squares")
    opt = {'tol': 1e-4, 'iter': 16}
    x_init = ImageData(geometry=ig)
    x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, DataContainer(sino_pad), opt=opt)
    plt.figure()
    plt.imshow(x_CGLS.array,vmin=0, vmax=1)
    plt.title('CGLS')
    plt.show()
##############################################################################
    # Saving images into fits using astrapy if required
    
    outfile = '{!s}_{!s}.fits'.format(filename,str(counter))
    im = x_CGLS.array
    add_val = np.min(im[:])
    im += abs(add_val)
    im = im/np.max(im[:])*65535
    hdu = fits.PrimaryHDU(np.uint16(im))
    hdu.writeto(outfile, overwrite=True)
    
    counter += 1
