from cil.framework import DataProcessor, ImageData, DataOrder
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D
import astra
from astra import astra_dict, algorithm, data3d
import numpy as np


class AstraBackProjector3D(DataProcessor):
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
        super(AstraBackProjector3D, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        self.vol_geom, self.proj_geom = convert_geometry_to_astra_vec_3D(self.volume_geometry, self.sinogram_geometry)
        
    
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
            data_temp = np.expand_dims(DATA.as_array(),axis=0)
        else:
            data_temp = DATA.as_array()

        if out is None:
            rec_id, arr_out = astra.create_backprojection3d_gpu(data_temp,
                            self.proj_geom,
                            self.vol_geom)
        else:
            if pad:
                arr_out = np.expand_dims(out.as_array(), axis=0)
            else:
                arr_out = out.as_array()
            rec_id = astra.data3d.link('-vol', self.vol_geom, arr_out)
            self.create_backprojection3d_gpu( data_temp, self.proj_geom, self.vol_geom, False, rec_id)

        # delete the GPU copy
        astra.data3d.delete(rec_id)
        
        if pad is True:
            arr_out = np.squeeze(arr_out, axis=0)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def create_backprojection3d_gpu(self, data, proj_geom, vol_geom, returnData=True, vol_id=None):
        """Create a backprojection of a sinogram (3D) using CUDA.
            :param data: Sinogram data or ID.
            :type data: :class:`numpy.ndarray` or :class:`int`
            :param proj_geom: Projection geometry.
            :type proj_geom: :class:`dict`
            :param vol_geom: Volume geometry.
            :type vol_geom: :class:`dict`
            :param returnData: If False, only return the ID of the backprojection.
            :type returnData: :class:`bool`
            :param vol_id: ID of the np array linked with astra
            :type vol_id: int, default None
            :returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the backprojection. Otherwise, returns a tuple containing the ID of the backprojection and the backprojection itself, in that order.
        """
        if isinstance(data, np.ndarray):
            sino_id = data3d.create('-sino', proj_geom, data)
        else:
            sino_id = data

        if vol_id is None:
            vol_id = data3d.create('-vol', vol_geom, 0)

        cfg = astra_dict('BP3D_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id
        alg_id = algorithm.create(cfg)
        algorithm.run(alg_id)
        algorithm.delete(alg_id)

        if isinstance(data, np.ndarray):
            data3d.delete(sino_id)

        if vol_id is not None:
            if returnData:
                return vol_id, data3d.get_shared(vol_id)
            else:
                return vol_id
        else:
            if returnData:
                return vol_id, data3d.get(vol_id)
            else:
                return vol_id
