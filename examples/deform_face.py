import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
from tfdeform.random_flows import random_deformation_momentum
from tfdeform.deform_util import dense_image_warp

face = misc.face() / 256
shape = face.shape
face = tf.convert_to_tensor(face[None, ...].astype('float32'))

offset = random_deformation_momentum(shape=[1, *shape[:2]], std=100.0, distance=50.0, stepsize=0.01)
face_deform = dense_image_warp(face, offset)

plt.figure()
plt.imshow(face[0, ...])
plt.show()

plt.figure()
plt.imshow(face_deform[0, ...])
plt.show()