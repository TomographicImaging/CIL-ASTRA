from ccpi.framework import  ImageData_cupy as ImageData

from ccpi.astra.processors import AstraBackProjector_cupy as AstraBackProjector

import astra
import cupy as cp

class AstraBackProjectorMC_cupy(AstraBackProjector):
    '''AstraBackProjector Multi channel
    
    Back project AcquisitionData to ImageData using ASTRA proj_id.
    
    Input: AcquisitionData
    Parameter: proj_id
    Output: ImageData
    '''
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3 or \
           dataset.number_of_dimensions == 4:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def process(self, out=None):
        
        DATA = self.get_input()
        IM = ImageData(geometry=self.volume_geometry)
        
        tmp_IM = cp.asnumpy(IM.as_array())
        tmp_DATA = cp.asnumpy(DATA.as_array())
        
        for k in range(IM.geometry.channels):
            rec_id, tmp_IM[k] = astra.create_backprojection(
                    tmp_DATA[k], 
                    self.proj_id)
            astra.data2d.delete(rec_id)
            
        IM.array = cp.array(tmp_IM)    
        
        if self.device == 'cpu':
            ret = IM
        else:
            scaling = self.volume_geometry.voxel_size_x**3  
            ret = scaling*IM
        
        if out is None:
            return ret
        else:
            out.fill(ret)
