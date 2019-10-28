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
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.astra.processors import AstraForwardProjector3D, AstraBackProjector3D
import numpy as np

class AstraProjector3DSimple(LinearOperator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        
        super(AstraProjector3DSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        # The order of the ouput sinogram is not the default one from the acquistion geometry
        # The order of the backprojection is the default one from the image geometry
                
        geomp.dimension_labels = ['vertical','angle','horizontal']
        geomp.shape = (geomp.pixel_num_v, len(geomp.angles), geomp.pixel_num_h)  
            
        self.sinogram_geometry = geomp 
        self.volume_geometry = geomv         
        
        self.fp = AstraForwardProjector3D(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        output_axes_order=['vertical','angle','horizontal'])
        
        self.bp = AstraBackProjector3D(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        output_axes_order=['vertical','horizontal_y','horizontal_x'])
                      
        # Initialise empty for singular value.
        self.s1 = None
    
    def direct(self, IM, out=None):
        self.fp.set_input(IM)
        
        if out is None:
            return self.fp.get_output()
        else:
            out.fill(self.fp.get_output())
    
    def adjoint(self, DATA, out=None):
        self.bp.set_input(DATA)
        
        if out is None:
            return self.bp.get_output()
        else:
            out.fill(self.bp.get_output())    
    
    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry 
                    
    def norm(self):
        x0 = self.volume_geometry.allocate('random')
        self.s1, sall, svec = LinearOperator.PowerMethod(self, 50, x0)
        return self.s1
    
    
if __name__  == '__main__':
        
    N = 30
    angles = np.linspace(0, np.pi, 180)
    ig = ImageGeometry(N, N, N)
    ag = AcquisitionGeometry('parallel','3D', angles, pixel_num_h = N, pixel_num_v=5)
    
    A = AstraProjector3DSimple(ig, ag)
    print(A.norm())    
    
    x = ig.allocate('random_int')
    sin = A.direct(x)
    
    y = ag.allocate('random_int')
    im = A.adjoint(y)
    
    
    
    