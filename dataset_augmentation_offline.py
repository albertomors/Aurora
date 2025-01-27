import numpy as np
import trimesh
import os
from scipy.spatial.transform import Rotation as R
import random

main_path = "your_location_of_modelnet40_auto_aligned"
save_path = "location_where_to_save"
target_shape=(32,32,32)


# Convert meshes into voxel grids which will be then saved as numpy arrays
def mesh_to_voxelgrid(mesh,target_shape=(32,32,32)):
    # Normalize the mesh to fit into a 28x28x28 voxel grid
    bbox = mesh.bounding_box.extents
    max_dimension = np.max(bbox)
    pitch = max_dimension / 28.0

    # Convert the mesh to a voxel grid with the calculated pitch
    voxel_grid = mesh.voxelized(pitch=pitch)

    # Extract the NumPy array from the voxel grid
    voxel_grid_array = voxel_grid.matrix
    
    # Adjust the shape of the voxel grid
    shape_diff = np.array(target_shape) - np.array(voxel_grid_array.shape)

    # If the grid is smaller than the target shape, pad it with zeros
    if np.any(shape_diff > 0):
        padding = [(max(0, d//2), max(0, d - d//2)) for d in shape_diff]
        voxel_grid_array = np.pad(voxel_grid_array, padding, mode='constant', constant_values=0)

    # If the grid is larger than the target shape, crop it to fit
    if np.any(shape_diff < 0):
        voxel_grid_array = voxel_grid_array[:target_shape(0), :target_shape(1), :target_shape(2)]

    return voxel_grid_array  # Ensure we return the correct array

# Rotate a mesh
def rotate_mesh(mesh, rotation_degrees=20, rotation_axis='z'):
    # Rotation matrix 
    rotation = R.from_euler(rotation_axis, rotation_degrees, degrees=True)
    rotation_matrix = rotation.as_matrix()
    # Apply rotation to the vertices of the mesh
    mesh.vertices = mesh.vertices.dot(rotation_matrix)
    return mesh

classes = os.listdir(main_path)

for class_name in classes: 

    for folder in ['train','test']:
        folder_path = os.path.join(main_path, class_name, folder)
        print('Currently on:', class_name, '-', folder)
        os.makedirs(os.path.join(save_path, class_name, folder), exist_ok=True)

        for filename in os.listdir(folder_path):  # Search for each file in the folder
            mesh = trimesh.load(os.path.join(folder_path, filename))
            numpy_voxel = mesh_to_voxelgrid(mesh)

            for i in range(18):
                new_mesh = rotate_mesh(mesh, i * 20 + random.randint(-2, 2))    # Generate rotated mesh with some noise in the rotation angle (+- 2 degrees)
                numpy_voxel = mesh_to_voxelgrid(new_mesh)
                '''  numpy_voxel = numpy_voxel.astype(int)  ''' # Save as 1/0 array instead of True/False array
                np.save(os.path.join(save_path, class_name, folder, f"{os.path.splitext(filename)[0]}_augmented_{i}.npy"), numpy_voxel) # Save the NumPy array to the specified save_path and add .npy extension














