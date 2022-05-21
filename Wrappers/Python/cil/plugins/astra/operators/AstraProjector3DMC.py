# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from cil.optimisation.operators import LinearOperator
from cil.plugins.astra.operators import AstraProjector3D
import numpy as np


class AstraProjector3DMC(LinearOperator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        super(AstraProjector3DMC, self).__init__(
            geomv, range_geometry=geomp
        )
        
        
        self.igtmp3D = self.domain_geometry().subset(channel=0)
        # self.igtmp3D.channels = 1
        # self.igtmp3D.shape = self.volume_geometry.shape[1:]
        # self.igtmp3D.dimension_labels = ['vertical', 'horizontal_y', 'horizontal_x']
        
        self.agtmp3D = self.range_geometry().subset(channel=0)
        # self.agtmp3D.channels = 1
        # self.agtmp3D.shape = self.sinogram_geometry.shape[1:]
        # self.agtmp3D.dimension_labels = ['vertical', 'angle', 'horizontal']      
        
        self.A3D = AstraProjector3DSimple(self.igtmp3D, self.agtmp3D)     

        self.channels = self.domain_geometry().channels
        
    def direct(self, x, out = None):
        
        if out is None:
            
            tmp = self.range_geometry().allocate()
            for k in range(self.channels):
                t = self.A3D.direct(x.subset(channel=k))            
                # this line is potentially leading to problems
                # as it is strongly linked to a DataContainer
                # wrapping NumPy arrays.
                np.copyto(tmp.array[k], t.array) 
            return tmp
        
        else:
            
            for k in range(self.channels):
                tmp = self.A3D.direct(x.subset(channel=k))
                # as comment above
                np.copyto(out.array[k], tmp.array)
        
    def adjoint(self, x, out = None):
        
        if out is None:
            
            tmp = self.domain_geometry().allocate()
            for k in range(self.channels):
                t = self.A3D.adjoint(x.subset(channel=k))            
                np.copyto(tmp.array[k], t.array) 
            return tmp
        
        else:
            
            for k in range(self.channels):
                tmp = self.A3D.adjoint(x.subset(channel=k))            
                np.copyto(out.array[k], tmp.array)         

    def calculate_norm(self):

        return self.A3D.norm()
    
    
    
if __name__  == '__main__':
    
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    
    N = 30
    channels = 5
    angles = np.linspace(0, np.pi, 180)
    ig = ImageGeometry(N, N, N, channels = channels)
    ag = AcquisitionGeometry('parallel','3D', angles, pixel_num_h = N, pixel_num_v=5, channels = channels)
    
    A3DMC = AstraProjector3DMC(ig, ag)
    z = A3DMC.norm()
    
#    igtmp3D = A3DMC.volume_geometry.clone()
#    igtmp3D.channels = 1
#    igtmp3D.shape = A3DMC.volume_geometry.shape[1:]
#    igtmp3D.dimension_labels = ['vertical', 'horizontal_y', 'horizontal_x']
#    
#    agtmp3D = A3DMC.sinogram_geometry.clone()
#    agtmp3D.channels = 1
#    agtmp3D.shape = A3DMC.sinogram_geometry.shape[1:]
#    agtmp3D.dimension_labels = ['vertical', 'angle', 'horizontal']      
#    
#    A3D = AstraProjector3DSimple(igtmp3D, agtmp3D)     
#    z = A3D.norm()   