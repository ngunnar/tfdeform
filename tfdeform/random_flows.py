import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops
import numpy as np
from tfdeform.deform_util import dense_image_warp
from tfdeform.convolve import gausssmooth


__all__ = ('random_deformation_linear',
           'random_deformation_momentum',
		   'random_deformation_momentum_sequence')


def image_gradients(image, mode='forward'):
    """Compute gradients of image."""
    if image.shape.ndims != 4:
        raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not %s.', image.shape)
    image_shape = array_ops.shape(image)
    batch_size, height, width, depth = array_ops.unstack(image_shape)
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]

    if mode == 'forward':
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
        dx = array_ops.reshape(dx, image_shape)
    else:
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([array_ops.zeros(shape, image.dtype), dy], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([array_ops.zeros(shape, image.dtype), dx], 2)
        dx = array_ops.reshape(dx, image_shape)

    return dy, dx

def jacobian(vf):
    """Compute the jacobian of a vectorfield pointwise."""
    vf0_dy, vf0_dx = image_gradients(vf[..., 0:1])
    vf1_dy, vf1_dx = image_gradients(vf[..., 1:2])

    r1 = tf.concat([vf0_dy[..., None], vf0_dx[..., None]], axis=-1)
    r2 = tf.concat([vf1_dy[..., None], vf1_dx[..., None]], axis=-1)

    return tf.concat([r1, r2], axis=-2)

def matmul(mat, vec):
    """Compute matrix @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 0, 1:2] * vec[..., 1:2]
    c21 = mat[..., 1, 0:1] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]

    return tf.concat([c11 + c12, c21 + c22],
                     axis=-1)

def matmul_transposed(mat, vec):
    """Compute matrix.T @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 1, 0:1] * vec[..., 1:2]
    c21 = mat[..., 0, 1:2] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]

    return tf.concat([c11 + c12, c21 + c22],
                     axis=-1)

def div(vf):
    """Compute divergence of vector field."""
    dy, _ = image_gradients(vf[..., 0:1], mode='backward')
    _, dx = image_gradients(vf[..., 1:2], mode='backward')
    return dx + dy

def batch_random_deformation_momentum_sequence(shape, std, distance, stepsize=0.1):
    r"""Create sequences of random diffeomorphic deformations.

    Parameters
    ----------
    shape : sequence of 3 ints
        Batch, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        Generated deformation field for each step (Batch, step, height, width)
    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """
    batch_size = shape[0]
    i = tf.constant(0, dtype=tf.int32)
    u0i, uji = random_deformation_momentum_sequence(shape[1:], std, distance, stepsize)
    def cond(i, u0i, uji):
        return i < batch_size - 1

    def body(i, u0i, uji):
        u1, u2 = random_deformation_momentum_sequence(shape[1:], std, distance, stepsize)
        u0i = tf.concat([u0i, u1[None,...]], axis=0)
        uji = tf.concat([uji, u2[None,...]], axis=0)
        print(i, u0i.shape, uji.shape)
        return i + 1, u0i, uji
    
    i, u0i, uji = tf.while_loop(
        cond, body, [i, u0i[None,...], uji[None,...]],
        shape_invariants=[0,
                          tf.TensorShape([batch_size, None, *shape[1:], 2]),
                          tf.TensorShape([batch_size, None, *shape[1:], 2])])
    return u0i, uji

def random_deformation_momentum_sequence(shape, std, distance, stepsize=0.1):
    r"""Create a sequence of random diffeomorphic deformations.

    Parameters
    ----------
    shape : sequence of 2 ints
        height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        Generated deformation field for each step (step, height, width)
    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """
    assert len(shape) == 2
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(shape[1]),
                                        math_ops.range(shape[0]))
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')

    base_coordinates = tf.concat([grid_y, grid_x], axis=-1)
    coordinates = tf.identity(base_coordinates)

    # Create mask to stop movement at edges
    mask = (tf.cos((grid_x - shape[1] / 2 + 1) * np.pi / (shape[1] + 2)) *
            tf.cos((grid_y - shape[0] / 2 + 1) * np.pi / (shape[0] + 2))) ** (0.25)

    # Total distance is given by std * n_steps * dt, we use this
    # to work out the exact numbers.
    n_steps = tf.cast(tf.math.ceil(distance / (std * stepsize)), 'int32')
    dt = distance / (tf.cast(n_steps, 'float32') * std)

    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std ** 2

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = dt * C * tf.random.normal(shape=[1, *shape, 2])

    # Using a while loop, generate the deformation step-by-step.
    def cond(i, from_coordinates, momenta):
        return i < n_steps

    def body(i, from_coordinates, momenta):
        v = mask * gausssmooth(momenta, std)

        d1 = matmul_transposed(jacobian(momenta), v)
        d2 = matmul(jacobian(v), momenta)
        d3 = div(v) * momenta
        f_c = from_coordinates[-1,...][None,...]
        momenta = momenta - dt * (d1 + d2 + d3)
        
        v = dense_image_warp(v, f_c - base_coordinates)
        f_c = dense_image_warp(f_c, v)
        
        from_coordinates = tf.concat([from_coordinates, f_c], axis=0)
        return i + 1, from_coordinates, momenta

    i = tf.constant(0, dtype=tf.int32)
    i, from_coordinates, momenta = tf.while_loop(
        cond, body, [i, coordinates, momenta],
        shape_invariants=[0,
                          tf.TensorShape([None,shape[0], shape[1], 2]),
                          0])

    from_total_offset = from_coordinates - base_coordinates
    from_total_diff = np.diff(np.concatenate([base_coordinates, from_coordinates], axis=0), axis=0)
    return from_total_offset, from_total_diff

def random_deformation_momentum(shape, std, distance, stepsize=0.1):
    r"""Create a random diffeomorphic deformation.

    Parameters
    ----------
    shape : sequence of 3 ints
        Batch, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        The end generated deformation field (Batch, height, width)

    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(shape[2]),
                                        math_ops.range(shape[1]))
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')

    base_coordinates = tf.concat([grid_y, grid_x], axis=-1)
    base_coordinates = tf.repeat(base_coordinates, shape[0], axis=0)
    coordinates = tf.identity(base_coordinates)

    # Create mask to stop movement at edges
    mask = (tf.cos((grid_x - shape[2] / 2 + 1) * np.pi / (shape[2] + 2)) *
            tf.cos((grid_y - shape[1] / 2 + 1) * np.pi / (shape[1] + 2))) ** (0.25)

    # Total distance is given by std * n_steps * dt, we use this
    # to work out the exact numbers.
    n_steps = tf.cast(tf.math.ceil(distance / (std * stepsize)), 'int32')
    dt = distance / (tf.cast(n_steps, 'float32') * std)

    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std ** 2

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = dt * C * tf.random.normal(shape=[*shape, 2])

    # Using a while loop, generate the deformation step-by-step.
    def cond(i, from_coordinates, momenta):
        return i < n_steps

    def body(i, from_coordinates, momenta):
        v = mask * gausssmooth(momenta, std)

        d1 = matmul_transposed(jacobian(momenta), v)
        d2 = matmul(jacobian(v), momenta)
        d3 = div(v) * momenta

        momenta = momenta - dt * (d1 + d2 + d3)
        v = dense_image_warp(v, from_coordinates - base_coordinates)
        from_coordinates = dense_image_warp(from_coordinates, v)

        return i + 1, from_coordinates, momenta

    i = tf.constant(0, dtype=tf.int32)
    i, from_coordinates, momenta = tf.while_loop(
        cond, body, [i, coordinates, momenta])

    from_total_offset = from_coordinates - base_coordinates

    return from_total_offset


def random_deformation_linear(shape, std, distance):
    r"""Create a random deformation.

    Parameters
    ----------
    shape : sequence of 3 ints
        Batch, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.

    Notes
    -----
    ``distance`` must be significantly smaller than ``std`` to guarantee that
    the deformation is smooth.
    """
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(shape[2]),
                                        math_ops.range(shape[1]))
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')

    # Create mask to stop movement at edges
    mask = (tf.cos((grid_x - shape[2] / 2 + 1) * np.pi / (shape[2] + 2)) *
            tf.cos((grid_y - shape[1] / 2 + 1) * np.pi / (shape[1] + 2))) ** (0.25)

    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = distance * C * tf.random.normal(shape=[*shape, 2])
    v = mask * gausssmooth(momenta, std)

    return v