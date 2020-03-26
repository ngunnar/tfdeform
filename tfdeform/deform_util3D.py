import numpy as np
import tensorflow as tf

__all__ = ('dense_image_warp',)

def _interpolate_trilinear(grid,
                          query_points,
                          name='interpolate_trilinear',
                          indexing='ijk'):
    """
    Finds values for query points on a grid using trilinear interpolation.
    Args:
        grid: a 5-D float `Tensor` of shape `[batch, depth, height, width, channels]`.
        query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 3]`.
        name: a name for the operation (optional).
        indexing: whether the query points are specified as row and column (ijk),
        or Cartesian coordinates (xyz).
    Returns:
        values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
        ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ijk' and indexing != 'xyz':
        raise ValueError('Indexing mode must be \'ijk\' or \'xyz\'')

    with tf.name_scope(name):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)
        shape = grid.shape
        if len(shape) != 5:
            msg = 'Grid must be 5 dimensional. Received size: '
            raise ValueError(msg + str(grid.shape))

        batch_size, depth, height, width, channels = shape
        query_type = query_points.dtype
        grid_type = grid.dtype

        if (len(query_points.shape) != 3 or query_points.shape[2] != 3):
            msg = ('Query points must be 3 dimensional and size 3 in dim 3. Received size: ')
            raise ValueError(msg + str(query_points.shape))

        _, num_queries, _ = query_points.shape

        if depth < 2 or height < 2 or width < 2:
            msg = 'Grid must be at least batch_size x 2 x 2 x 2 in size. Received size: '
            raise ValueError(msg + str(grid.shape))

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1, 2] if indexing == 'ijk' else [2, 1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2)

        for dim in index_order:
            with tf.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.minimum(tf.maximum(min_floor, tf.floor(queries)), max_floor)
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

        if batch_size * depth * height * width > np.iinfo(np.int32).max / 8:
            error_msg = """The image size or batch size is sufficiently large
                         that the linearized addresses used by tf.gather
                         may exceed the int32 limit."""
            raise ValueError(error_msg)

        flattened_grid = tf.reshape(grid, [batch_size * depth * height * width, channels])
        batch_offsets = tf.reshape(tf.range(batch_size) * depth * height * width, [batch_size, 1])

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(z_coords, y_coords, x_coords, name):
            with tf.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + z_coords * height * width + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 8 corners around each query point
        c000 = gather(floors[0], floors[1], floors[2], 'c000')
        c001 = gather(floors[0], floors[1], ceils[2], 'c001')
        c010 = gather(floors[0], ceils[1], floors[2], 'c010')
        c011 = gather(floors[0], ceils[1], ceils[2], 'c011')
        c100 = gather(ceils[0], floors[1], floors[2], 'c100')
        c101 = gather(ceils[0], floors[1], ceils[2], 'c101')
        c110 = gather(ceils[0], ceils[1], floors[2], 'c110')
        c111 = gather(ceils[0], ceils[1], ceils[2], 'c111')
    
        # now, do the actual interpolation
        with tf.name_scope('interpolate'):
            c00 = c000 * (1 - alphas[2]) + c100*alphas[2]
            c01 = c001 * (1 - alphas[2]) + c101*alphas[2]
            c10 = c010 * (1 - alphas[2]) + c110*alphas[2]
            c11 = c011 * (1 - alphas[2]) + c111*alphas[2]

            c0 = c00 * (1 - alphas[1]) + c10 * alphas[1]
            c1 = c01 * (1 - alphas[1]) + c11 * alphas[1]

            interp = c0*(1 - alphas[0]) + c1*alphas[0]
        return interp

def dense_image_warp(image, flow, name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, k, j, i, c] is
    images[b, k - flow[b,k,j,i,0], j - flow[b, k, j, i, 1], i - flow[b, k, j, i, 2], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by trilinear
    interpolation of the 8 nearest pixels around
    (b, k - flow[b, k, j, i, 0], j - flow[b, k, j, i, 1], i - flow[b, k, j, i, 2]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    Args:
        image: 5-D float `Tensor` with shape `[batch, depth, height, width, channels]`.
        flow: A 5-D float `Tensor` with shape `[batch, batch, height, width, 3]`.
        name: A name for the operation (optional).
        Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
        and do not necessarily have to be the same type.
    Returns:
        A 5-D float `Tensor` with shape`[batch, depth, height, width, channels]`
          and same type as input image.
      Raises:
        ValueError: if depth < 2 or height < 2 or width < 2 or the inputs have the wrong number
                    of dimensions.
    """
    with tf.name_scope(name):
        batch_size, depth, height, width, channels = image.shape
        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        
        grid_x, grid_y, grid_z = tf.meshgrid(tf.range(depth), 
                                            tf.range(height),
                                            tf.range(width),
                                            indexing ='ij')
        stacked_grid = tf.cast(tf.stack([grid_x, grid_y, grid_z], axis=3), flow.dtype)
    
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, depth * height * width, 3])
        # Compute values at the query points, then reshape the result back to the
        # image grid.    
        interpolated = _interpolate_trilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, depth, height, width, channels])
        return interpolated