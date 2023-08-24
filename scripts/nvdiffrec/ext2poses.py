import numpy as np

# set of variables
extrinsics_file = "380extrinsics.npy"
poses_file = "poses_bounds.npy"
x_focal = 8.1918643e+03
y_focal = 8.1916558e+03

# load the extrinsics
matrix = np.load(extrinsics_file, allow_pickle = True)[()]
extrinsics = []

for key in sorted(matrix):
    extrinsics.append(matrix[key])
    
extrinsics = np.stack(extrinsics, axis=0)
rows = len(extrinsics)

# change axis/columns from [x -y -z] to [-y x z]
poses = np.concatenate([extrinsics[:, :, 1:2], extrinsics[:, :, 0:1], -extrinsics[:, :, 2:3], extrinsics[:, :, 3:4]], 2)

# remove 4th row
poses = np.delete(poses, 3, axis=1)

# Calculate max zs and min zs // trial
zs = np.concatenate([2 - poses[:,:,3], 2 - poses[:,2,:3]], 1)
max_zs = zs.max(axis = 1)
min_zs = np.absolute(zs.min(axis = 1))

# Generate 5th columns
focal_len = (x_focal + y_focal) / 2.0
col_5 = np.empty([rows, 3, 1])
for x in range(len(col_5)):
        col_5[x][0][0] = 3208
        col_5[x][1][0] = 2200
        col_5[x][2][0] = focal_len # = x_focal + y_focal / 2 , y_focal = 8.1899722e+03 = focal length from intrinsics 
        # col_5[x][2][0] = 8.14945237e+03 # focal length from colmap

# Append 5th column to the matrix
poses = np.append(poses, col_5, axis=2)

# flatten the matrix
poses = poses.reshape(rows, 15)

# Extract shortest and furthest points from colmap matrix and append them to matrix
# poses_tmp = np.load("850_poses_bounds.npy")
col_16 = np.empty([rows, 1])
col_17 = np.empty([rows, 1])

# for x in range(rows):
#     col_16[x][0] = poses_tmp[x][15]
#     col_17[x][0] = poses_tmp[x][16]
    
# poses = np.append(poses, col_16, axis=1)
# poses = np.append(poses, col_17, axis=1)
    
# use zs instead
for x in range(rows):
    col_16[x][0] = min_zs[x]
    col_17[x][0] = max_zs[x]
poses = np.append(poses, col_16, axis=1)
poses = np.append(poses, col_17, axis=1)

# Save file
np.save(poses_file, poses)