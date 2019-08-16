# -*- coding: utf-8 -*-
#    This work is independent part of the Core Imaging Library developed by
#    Visual Analytics and Imaging System Group of the Science Technology
#    Facilities Council, STFC
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ccpi.optimisation.operators import LinearOperator
from ccpi.astra.operators import AstraProjector3DSimple
import numpy as np


class AstraProjector3DMC(LinearOperator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        super(AstraProjector3DMC, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.igtmp3D = self.volume_geometry.clone()
        self.igtmp3D.channels = 1
        self.igtmp3D.shape = self.volume_geometry.shape[1:]
        self.igtmp3D.dimension_labels = ['vertical', 'horizontal_y', 'horizontal_x']
        
        self.agtmp3D = self.sinogram_geometry.clone()
        self.agtmp3D.channels = 1
        self.agtmp3D.shape = self.sinogram_geometry.shape[1:]
        self.agtmp3D.dimension_labels = ['vertical', 'angle', 'horizontal']      
        
        self.A3D = AstraProjector3DSimple(self.igtmp3D, self.agtmp3D)     
            
        self.s1 = None
        self.channels = self.volume_geometry.channels
        
    def direct(self, x, out = None):
        
        if out is None:
            
            tmp = self.sinogram_geometry.allocate()
            for k in range(self.channels):
                t = self.A3D.direct(x.subset(channel=k))            
                np.copyto(tmp.array[k], t.array) 
            return tmp
        
        else:
            
            for k in range(self.channels):
                tmp = self.A3D.direct(x.subset(channel=k))            
                np.copyto(out.array[k], tmp.array)                             
        
    def adjoint(self, x, out = None):
        
        if out is None:
            
            tmp = self.volume_geometry.allocate()
            for k in range(self.channels):
                t = self.A3D.adjoint(x.subset(channel=k))            
                np.copyto(tmp.array[k], t.array) 
            return tmp
        
        else:
            
            for k in range(self.channels):
                tmp = self.A3D.adjoint(x.subset(channel=k))            
                np.copyto(out.array[k], tmp.array)         
 
    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry  
    
    def calculate_norm(self):

        return self.A3D.norm()
    
    
    
if __name__  == '__main__':
    
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    import numpy as np
    
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