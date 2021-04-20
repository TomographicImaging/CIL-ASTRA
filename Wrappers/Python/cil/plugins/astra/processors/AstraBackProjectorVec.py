from cil.framework import DataProcessor, ImageData, DataOrder
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec
import astra
import numpy


class AstraBackProjectorVec(DataProcessor):
    '''AstraBackProjector3D
    
    Back project AcquisitionData to ImageData using ASTRA proj_geom, vol_geom.
    
    Input: AcquisitionData
    Parameter: proj_geom, vol_geom
    Output: ImageData
    '''

    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom}
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraBackProjectorVec, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        self.vol_geom, self.proj_geom = convert_geometry_to_astra_vec(self.volume_geometry, self.sinogram_geometry)
        
    
    def check_input(self, dataset):

        if self.sinogram_geometry.shape != dataset.geometry.shape:
            raise ValueError("Dataset not compatible with geometry used to create the projector")  
    
        return True
    
    def set_ImageGeometry(self, volume_geometry):

        DataOrder.check_order_for_engine('astra', volume_geometry)

        if len(volume_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(volume_geometry.number_of_dimensions))  

        self.volume_geometry = volume_geometry.copy()

    def set_AcquisitionGeometry(self, sinogram_geometry):

        DataOrder.check_order_for_engine('astra', sinogram_geometry)

        if len(sinogram_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(volume_geometry.number_of_dimensions))  

        self.sinogram_geometry = sinogram_geometry.copy()

    def process(self, out=None):

        DATA = self.get_input()

        pad = False
        if len(DATA.shape) == 2:
            #for 2D cases
            pad = True
            data_temp = numpy.expand_dims(DATA.as_array(),axis=0)
        else:
            data_temp = DATA.as_array()

        rec_id, arr_out = astra.create_backprojection3d_gpu(data_temp,
                            self.proj_geom,
                            self.vol_geom)

        astra.data3d.delete(rec_id)
        
        if pad is True:
            arr_out = numpy.squeeze(arr_out, axis=0)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
            