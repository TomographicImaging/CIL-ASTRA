#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from ccpi.optimisation.operators import LinearOperator, ChannelwiseOperator
from ccpi.astra.operators import AstraProjectorFlexible
from ccpi.astra.operators import AstraProjectorSimple

class AstraOperator(LinearOperator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, device='gpu'):
        
        super(AstraOperator, self).__init__(domain_geometry=geomv, range_geometry=geomp)

        self.volume_geometry = geomv
        self.sinogram_geometry = geomp

        sinogram_geometry_sc = geomp.subset(channel=0)
        volume_geometry_sc = geomv.subset(channel=0)

        if device is 'gpu':
            operator = AstraProjectorFlexible(volume_geometry_sc, sinogram_geometry_sc)
        else:
            UserWarning("ASTRA projectors running on CPU will not make use of enhanced geometry parameters")
            if self.sinogram_geometry.dimension == '2D':
                operator = AstraProjectorSimple(volume_geometry_sc, sinogram_geometry_sc,  device='cpu') 
            else:
                raise NotImplementedError("Cannot process 3D data without a GPU")

        if geomp.channels > 1: 
            operator_full = ChannelwiseOperator(operator, self.sinogram_geometry.channels, dimension='prepend')
            self.operator = operator_full
        else:
            self.operator = operator

    def direct(self, IM, out=None):
        return self.operator.direct(IM, out=out)
    
    def adjoint(self, DATA, out=None):
        return self.operator.adjoint(DATA, out=out)

    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry 

if __name__  == '__main__':
    
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    import numpy as np
    
    N = 30
    angles = np.linspace(0, np.pi, 180)
    ig = ImageGeometry(N, N)
    ag = AcquisitionGeometry('parallel','2D', angles, pixel_num_h=N)
    A = AstraOperator(ig, ag, 'cpu')
    print(A.norm())

    ig = ImageGeometry(N, N, N)
    ag = AcquisitionGeometry('parallel','3D', angles, pixel_num_v=N, pixel_num_h=N,dimension_labels=['vertical', 'angle', 'horizontal'])
    A = AstraOperator(ig, ag, 'gpu')
    print(A.norm())

    ig = ImageGeometry(N, N, channels=2)
    ag = AcquisitionGeometry('parallel','2D', angles, pixel_num_h=N, channels= 2)
    A = AstraOperator(ig, ag, 'cpu')
    print(A.norm())

    ig = ImageGeometry(N, N, N, channels=2)
    ag = AcquisitionGeometry('parallel','3D', angles, pixel_num_v=N, pixel_num_h=N, channels=2,dimension_labels=['vertical', 'angle', 'horizontal'])
    A = AstraOperator(ig, ag, 'gpu')
    print(A.norm())
