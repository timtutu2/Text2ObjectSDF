import numpy as np
from skimage.measure import marching_cubes

data = np.load("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/19d3ba04e165e67dcb4387db711dc078_sdf_tau0p1.npz")

vol = data["sdf_grid"]
verts, faces, normals, _ = marching_cubes(vol, level=0)

import trimesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
mesh.show()