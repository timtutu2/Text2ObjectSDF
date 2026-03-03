import numpy as np
from skimage.measure import marching_cubes
import nrrd

data = np.load("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/sampling.npz")

data_nrrd, header = nrrd.read("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/model_4.nrrd")

vol = data["sdf_grid"]
print(header.get("space directions"))
print(data.files)
# verts, faces, normals, _ = marching_cubes(vol, level=0)

# import trimesh
# mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
# mesh.show()