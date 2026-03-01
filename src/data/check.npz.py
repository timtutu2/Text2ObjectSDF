import numpy as np

# Path to your SDF npz file
file_path = "/home/tim/Desktop/UCSD/ECE285/Text2ObjectSDF/sampling/sdf_data_1.npz" 

# Load npz file
data = np.load(file_path)

# Check available keys
print("Keys in file:", data.files)

# Compute mean of sdf
if 'sdf' in data:
    print("Mean of sdf:", data['sdf'].mean())
else:
    print("'sdf' key not found in this npz file.")