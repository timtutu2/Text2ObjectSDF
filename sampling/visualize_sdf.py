import numpy as np
from skimage.measure import marching_cubes
import nrrd
import trimesh

data = np.load("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/sampling.npz")

data_nrrd, header = nrrd.read("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/model_4.nrrd")

vol = data["sdf_grid"]
print(header.get("space directions"))
print(data.files)

###############################
# Check the data
print(data.files)
print("spacing:", data["spacing"])
print("tau:", data["tau"])
print("sdf_grid shape:", data["sdf_grid"].shape)

###############################
# Visualize the data
# vol = data["sdf_grid"]
# spacing = tuple(data["spacing"].tolist())

# verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=spacing)
# mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
# mesh.show()

###############################
sdf = data["sdf_grid"]
print("neg ratio:", (sdf < 0).mean(), "pos ratio:", (sdf > 0).mean())
print("min/max:", sdf.min(), sdf.max())
