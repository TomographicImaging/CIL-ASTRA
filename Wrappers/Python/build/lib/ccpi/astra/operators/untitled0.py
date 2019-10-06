#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:36:44 2019

@author: vaggelis
"""

from ccpi.optimisation.operators import LinearOperator
from ccpi.astra.operators import AstraProjectorSimple, AstraProjectorMC, AstraProjector3DSimple, AstraProjector3DMC

from ccpi.framework import ImageGeometry, AcquisitionGeometry
import numpy as np
    

# test 3D AstraOperator
N = 30
angles = np.linspace(0, np.pi, 180)
ig3D = ImageGeometry(N, N, N)
ag3D = AcquisitionGeometry('parallel','3D', angles, pixel_num_h = N, pixel_num_v=5)

print(ag3D.dimension_labels, ag3D.shape)

A3D = AstraProjector3DSimple(ig3D, ag3D)

print(A3D.sinogram_geometry.dimension_labels, A3D.sinogram_geometry.shape)

z3D = A3D.norm()
print(' Astra operator norm for 3D = {}'.format(z3D))

# test 2D AstraOperator
N = 30
angles = np.linspace(0, np.pi, 180)
ig2D = ImageGeometry(N, N)
ag2D = AcquisitionGeometry('parallel','2D', angles, pixel_num_h = N)

print(ag2D.dimension_labels, ag2D.shape)

A2D = AstraProjectorSimple(ig2D, ag2D, device = 'cpu')

print(A2D.sinogram_geometry.dimension_labels, A2D.sinogram_geometry.shape)

z2D = A2D.norm()
print(' Astra operator norm for 2D = {}'.format(z2D))

# test 3DMC AstraOperator
N = 30
channels = 5
angles = np.linspace(0, np.pi, 180)
ig3DMC = ImageGeometry(N, N, N, channels = channels)
ag3DMC = AcquisitionGeometry('parallel','3D', angles, pixel_num_h = N, pixel_num_v=5, channels = channels)

A3DMC = AstraProjector3DMC(ig3DMC, ag3DMC)
z3DMC = A3DMC.norm()

print(' Astra operator norm for 3DMC = {}'.format(z3DMC))
print(ag3DMC.shape, A3DMC.A3D.sinogram_geometry.shape)

