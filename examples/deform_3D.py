from tfdeform.random_flows3D import random_deformation_momentum
from tfdeform.deform_util3D import dense_image_warp
import numpy as np
import plotly.graph_objects as go

data = np.zeros((20,20,20))
data[7:15,7:15,7:15] = 1

shape = [1,*data.shape]
std = 15
distance = 20
stepsize = 1
flow = random_deformation_momentum(shape, std, distance, stepsize)
warped = dense_image_warp(data[None,..., None], flow)

X, Y, Z = np.mgrid[0:20:1, 0:20:1, 0:20:1]

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=data.flatten(),
    isomin=0.1,
    isomax=1.0,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=100, # needs to be a large number for good volume rendering
    ))
fig.show()

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=warped[0,:,:,:,0].numpy().flatten(),
    isomin=0.1,
    isomax=1.0,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=100, # needs to be a large number for good volume rendering
    ))
fig.show()