import astra
import numpy

def convert_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in):

    '''Set up ASTRA Volume and projection geometry, not stored

       :param volume_geometry: ccpi.framework.ImageGeometry
       :param sinogram_geometry: ccpi.framework.AcquisitionGeometry
       
       :returns ASTRA volume and sinogram geometry'''
 
    sinogram_geometry = sinogram_geometry_in.copy()
    sinogram_geometry.config.system.update_reference_frame()

    angles = sinogram_geometry.config.angles
    system = sinogram_geometry.config.system
    panel = sinogram_geometry.config.panel

    #get units
    degrees = angles.angle_unit == sinogram_geometry.DEGREE
    
    if sinogram_geometry.dimension == '2D':
        #create a 3D astra geom from 2D CIL geometry
        volume_geometry_temp = volume_geometry.copy()
        volume_geometry_temp.voxel_num_z = 1
        volume_geometry_temp.voxel_size_z = volume_geometry_temp.voxel_size_x
            
        if panel.pixel_size[1] != panel.pixel_size[0]:
            panel.pixel_size[1] =  panel.pixel_size[0]

        row = numpy.zeros((3,1))
        row[0] = panel.pixel_size[0] * system.detector.direction_x[0]
        row[1] = panel.pixel_size[0] * system.detector.direction_x[1]

        if 'right' in panel.origin:
            row *= -1

        col = numpy.zeros((3,1))
        col[2] = panel.pixel_size[1]

        det = numpy.zeros((3,1))
        det[0] = system.detector.position[0]
        det[1] = system.detector.position[1]

        src = numpy.zeros((3,1))
        if sinogram_geometry.geom_type == 'parallel':
            src[0] = system.ray.direction[0]
            src[1] = system.ray.direction[1]
            projector = 'parallel3d_vec'
        else:
            src[0] = system.source.position[0]
            src[1] = system.source.position[1]
            projector = 'cone_vec'

    else:
        volume_geometry_temp = volume_geometry.copy()
 
        row = panel.pixel_size[0] * system.detector.direction_x.reshape(3,1)
        col = panel.pixel_size[1] * system.detector.direction_y.reshape(3,1)
        det = system.detector.position.reshape(3, 1)

        if 'right' in panel.origin:
            row *= -1
        if 'top' in panel.origin:
            col *= -1

        if sinogram_geometry.geom_type == 'parallel':
            src = system.ray.direction.reshape(3,1)
            projector = 'parallel3d_vec'
        else:
            src = system.source.position.reshape(3,1)
            projector = 'cone_vec'

    #Build for astra 3D only
    vectors = numpy.zeros((angles.num_positions, 12))

    for i, theta in enumerate(angles.angle_data):
        ang = - angles.initial_angle - theta

        rotation_matrix = rotation_matrix_z_from_euler(ang, degrees=degrees)

        vectors[i, :3]  = rotation_matrix.dot(src).reshape(3)
        vectors[i, 3:6] = rotation_matrix.dot(det).reshape(3)
        vectors[i, 6:9] = rotation_matrix.dot(row).reshape(3)
        vectors[i, 9:]  = rotation_matrix.dot(col).reshape(3)
    
    proj_geom = astra.creators.create_proj_geom(projector, panel.num_pixels[1], panel.num_pixels[0], vectors)    
    vol_geom = astra.create_vol_geom(volume_geometry_temp.voxel_num_y,
                                    volume_geometry_temp.voxel_num_x,
                                    volume_geometry_temp.voxel_num_z,
                                    volume_geometry_temp.get_min_x(),
                                    volume_geometry_temp.get_max_x(),
                                    volume_geometry_temp.get_min_y(),
                                    volume_geometry_temp.get_max_y(),
                                    volume_geometry_temp.get_min_z(),
                                    volume_geometry_temp.get_max_z()
                                    )


    return vol_geom, proj_geom

def rotation_matrix_z_from_euler(angle, degrees):
    '''Returns 3D rotation matrix for z axis using direction cosine

    :param angle: angle or rotation around z axis
    :type angle: float
    :param degrees: if radian or degrees
    :type bool: defines the unit measure of the angle
    '''
    if degrees:
        alpha = angle / 180. * numpy.pi
    else:
        alpha = angle

    rot_matrix = numpy.zeros((3,3), dtype=numpy.float64)
    rot_matrix[0][0] = numpy.cos(alpha)
    rot_matrix[0][1] = - numpy.sin(alpha)
    rot_matrix[1][0] = numpy.sin(alpha)
    rot_matrix[1][1] = numpy.cos(alpha)
    rot_matrix[2][2] = 1
    
    return rot_matrix
