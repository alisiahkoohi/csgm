import os
import h5py


def save_exp_to_h5(path, args, **kwargs):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Check if the file already exists. If it does, delete it.
    if os.path.exists(path):
        os.remove(path)

    # HDF5 File.
    file = h5py.File(path, 'a')
    for key, value in vars(args).items():
        # Add the key and value to the file.
        file[key] = value
    for key, value in kwargs.items():
        if isinstance(value, dict):
            group = file.require_group(key)
            for k, v in value.items():
                group[str(k)] = v
        else:
            file[key] = value
    file.close()


def load_exp_from_h5(path, *args):
    """
    Load data from an HDF5 file using specified keys.

    Args:
        path (str): Path to the HDF5 file.
        *args (str): Keys for the data to be loaded.

    Returns:
        dict: A dictionary containing the loaded data.
    """
    # Open the HDF5 file.
    with h5py.File(path, 'r') as file:
        # Create an empty dictionary to store the loaded data.
        data_dict = {}
        # Load the specified data and add it to the dictionary.
        for key in args:
            if isinstance(file[key], h5py._hl.group.Group): # pylint: disable=protected-access
                inner_dict = {}
                for k in file[key].keys():
                    inner_dict[k] = file[key][k][...]
                data_dict[key] = inner_dict
            else:
                data_dict[key] = file[key][...]
    # Return the dictionary containing the loaded data.
    return data_dict
