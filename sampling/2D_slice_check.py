import matplotlib.pyplot as plt
import numpy as np

data = np.load("/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/sdf_data_4.npz")
sdf = data["sdf_grid"]

save_path = "/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/2D_slice_check.png"
plt.imshow(sdf[:, :, sdf.shape[2]//2])
plt.colorbar()
plt.savefig(save_path)
plt.close()