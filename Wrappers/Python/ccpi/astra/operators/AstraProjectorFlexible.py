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

from ccpi.optimisation.operators import LinearOperator
from ccpi.astra.processors import AstraForwardProjectorVec, AstraBackProjectorVec



class AstraProjectorFlexible(LinearOperator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        
        super(AstraProjectorFlexible, self).__init__(domain_geometry=geomv, range_geometry=geomp)
                    
        self.sinogram_geometry = geomp 
        self.volume_geometry = geomv         
        
        self.fp = AstraForwardProjectorVec(volume_geometry=geomv, sinogram_geometry=geomp)
            
        self.bp = AstraBackProjectorVec(volume_geometry=geomv, sinogram_geometry=geomp)
              
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
        s1, sall, svec = LinearOperator.PowerMethod(self, 50, x0)
        return s1

if __name__  == '__main__':

    N = 30
    angles = np.linspace(0, np.pi, 180)
    ig = ImageGeometry(N, N, N)
    ag = AcquisitionGeometry('parallel','3D', angles, pixel_num_h = N, pixel_num_v=5)

    A = AstraProjectorFlexible(ig, ag)
    print(A.norm())    

    x = ig.allocate('random_int')
    sin = A.direct(x)

    y = ag.allocate('random_int')
    im = A.adjoint(y)