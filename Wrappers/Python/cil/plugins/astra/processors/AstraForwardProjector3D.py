from cil.framework import DataProcessor, AcquisitionData
from cil.plugins.astra.utilities import convert_geometry_to_astra
import astra

class AstraForwardProjector3D(DataProcessor):
    '''AstraForwardProjector3D
    
    Forward project ImageData to AcquisitionData using ASTRA proj_geom and 
    vol_geom.
    
    Input: ImageData
    Parameter: proj_geom, vol_geom
    Output: AcquisitionData
    '''
    ASTRA_LABELS_VOL = ['vertical','horizontal_y','horizontal_x']
    ASTRA_LABELS_PROJ = ['vertical','angle','horizontal']

    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom,
                  }
        
        super(AstraForwardProjector3D, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        self.vol_geom, self.proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
         
    def check_input(self, dataset):
        if dataset.number_of_dimensions != 3:
            raise ValueError("Expected input dimensions 3, got {0}"\
                             .format(dataset.number_of_dimensions))

        order = [dataset.dimension_labels[0],dataset.dimension_labels[1],dataset.dimension_labels[2]]
        if order != AstraForwardProjector3D.ASTRA_LABELS_VOL:
            raise ValueError("Acquistion geometry expects dimension label order {0} for ASTRA compatibility got {1}".format(AstraForwardProjector3D.ASTRA_LABELS_PROJ_3D, order))  

        return True       
    
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry.copy()
        self.volume_geometry.dimension_labels = AstraForwardProjector3D.ASTRA_LABELS_VOL

    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry.copy()
        self.sinogram_geometry.dimension_labels = AstraForwardProjector3D.ASTRA_LABELS_PROJ
    
    def set_vol_geom(self, vol_geom):
        self.vol_geom = vol_geom
    
    def process(self, out=None):
        
        IM = self.get_input()

        sinogram_id, arr_out = astra.create_sino3d_gpu(IM.as_array(), self.proj_geom, self.vol_geom)
        astra.data3d.delete(sinogram_id)
        
        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self.sinogram_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
