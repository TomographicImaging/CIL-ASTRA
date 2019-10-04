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
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector, AstraFilteredBackProjector

class AstraProjectorSimple(LinearOperator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjector(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id = None,
                                        device=device)
        
        self.bp = AstraBackProjector(volume_geometry = geomv,
                                        sinogram_geometry = geomp,
                                        proj_id = None,
                                        device = device)
        
        self.fbp = AstraFilteredBackProjector(volume_geometry = geomv,
                                        sinogram_geometry = geomp,
                                        proj_id = None,
                                        filter_type = None,
                                        device=device)        
         
        
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
    
    def FBP(self, DATA, filter_type):
        
        self.fbp.filter_type = filter_type
        self.fbp.set_input(DATA)
        out = self.fbp.get_output()
        
        return out    
    
    def norm(self):
        x0 = self.volume_geometry.allocate('random')
        self.s1, sall, svec = LinearOperator.PowerMethod(self, 50, x0)
        return self.s1
