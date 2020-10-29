from cil.framework import  ImageData


from cil.plugins.astra.processors import AstraBackProjector

import astra

class AstraBackProjectorMC(AstraBackProjector):
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
        
        if out is None:
            IM = self.volume_geometry.allocate(None)
        else:
            IM = out

        for k in range(IM.geometry.channels):
            data_temp = DATA.as_array()[k]
            rec_id, IM.as_array()[k] = astra.create_backprojection(data_temp, self.proj_id)     
            astra.data2d.delete(rec_id)
             
        if out is None:
            return IM   
