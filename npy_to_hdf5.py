import os
import numpy as np
import h5py


data_dir = 'path_location_folder_of_random_oriented_dataset'

def npy_to_hdf5(data_dir, hdf5_filename):
    """
    Converts .npy files in data_dir to a single HDF5 file.
    
    Parameters:
    - data_dir: Path to the directory containing .npy files structured by class.
    - hdf5_filename: The name of the HDF5 file to save.
    """
    # Create the HDF5 file
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        classes = sorted(os.listdir(data_dir))
        class_group = hdf5_file.create_group("classes")  # Group to store class data

        for class_idx, class_name in enumerate(classes):
            print('class_name:',class_name)
            class_path = os.path.join(data_dir, class_name)

            if not os.path.isdir(class_path):
                continue  # Skip non-directory items

            # Create a group for each class
            class_group.create_group(class_name)

            # Process the 'train' and 'test' folders inside each class
            for split in ['train', 'test']:
                print('split:',split)
                split_path = os.path.join(class_path, split)

                if not os.path.isdir(split_path):
                    continue  # Skip if no 'train' or 'test' folder exists

                # Create group for train/test split inside the class
                split_group = class_group[class_name].create_group(split)

                # Process .npy files in the 'train' or 'test' folder
                npy_files = [f for f in os.listdir(split_path) if f.endswith('.npy')]

                for npy_file in npy_files:
                    file_path = os.path.join(split_path, npy_file)

                    try:
                        # Load .npy file
                        voxel_data = np.load(file_path)

                        # Save the numpy array as a dataset in HDF5 file
                        dataset_name = npy_file[:-4]  # Use file name without extension
                        split_group.create_dataset(dataset_name, data=voxel_data)

                        print(f"Converted {npy_file} for class {class_name}/{split}.")

                    except Exception as e:
                        print(f"Failed to process {npy_file} in {class_name}/{split}. Error: {e}")

    print(f"Conversion complete: {hdf5_filename}")


hdf5_output_file = "dataset_random_orient_40_32.h5"

npy_to_hdf5(data_dir, hdf5_output_file)




