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
from ccpi.astra.processors import AstraForwardProjectorMC, AstraBackProjectorMC
from ccpi.astra.operators import AstraProjectorSimple     

class AstraProjectorMC(LinearOperator):
    """ASTRA Multichannel projector"""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorMC, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjectorMC(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id=None,
                                        device=device)
        
        self.bp = AstraBackProjectorMC(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id=None,
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
    
    def calculate_norm(self):
        
        igtmp = self.volume_geometry.clone()
        igtmp.shape = self.volume_geometry.shape[1:]
        igtmp.dimension_labels = ['horizontal_y', 'horizontal_x']
        igtmp.channels = 1

        agtmp = self.sinogram_geometry.clone()
        agtmp.shape = self.sinogram_geometry.shape[1:]
        agtmp.dimension_labels = ['angle', 'horizontal']
        agtmp.channels = 1        
        
        Atmp = AstraProjectorSimple(igtmp, agtmp, device = 'gpu')
              
        #TODO Approach with clone should be better but it doesn't work atm
        
        #igtmp = self.volume_geometry.clone()
        #agtmp = self.sinogram_geometry.clone()
        #igtmp.channels=1
        #agtmp.channels=1
        #igtmp.dimension_labels = ['angle','vertical']
        #agtmp.dimension_labels = ['angle','vertical']
        #Atmp = AstraProjectorSimple(igtmp, agtmp, self.fp.device)
        
        
        return Atmp.norm()    
    

