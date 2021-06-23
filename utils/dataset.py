import torch
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import PosixPath
from typing import List, Tuple
import h5py
from torch.utils.data import Dataset


class Dataset():
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask


    """
    def __init__(
            self,
            images_dir,
            element='rain'
    ):

        self.ids = pd.read_csv(images_dir,header=None)
        # print(self.ids)
        self.images_fps = self.ids.iloc[:,0].values
        # print(self.images_fps)
        self.masks_fps = self.ids.iloc[:,1].values
        # print(self.masks_fps)
        self.element=element
        # convert str names to class values on masks

    def __getitem__(self, i):

        # read data
        # print(image.shape)
        # image = (image-10)/(60-10)
        # print(image)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = xr.open_dataset(self.masks_fps[i])[self.element].values[:,:-11,:-11]
        mask = np.nan_to_num(mask)
        # mask = np.where(mask>=0.1, 1, 0)
        image = np.nan_to_num(xr.open_mfdataset(eval(self.images_fps[i]), concat_dim='time', combine='nested').REF.values.squeeze().swapaxes(0,3).swapaxes(1,3).swapaxes(2,3).swapaxes(0,1))[-10:,...,:-11,:-11]
        print(image.shape)
        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
        #return minmaxscaler(image), mask
        # return image, mask
        # print(mask)

        # apply augmentations
        #if self.augmentation:
        #    sample = self.augmentation(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']

        # apply preprocessing
        #if self.preprocessing:
        #    sample = self.preprocessing(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']


    def __len__(self):
        return len(self.ids)


class DatasetH5(Dataset):
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    """

    def __init__(
        self,
        h5_file: PosixPath,
        element = 'rain'
    ):
        self.h5_file = h5_file

        # Placeholder for catchment attributes stats
        self.df = None
        self.element = element
        (self.x, self.y) = self._preload_data()

        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return {'image':x, 'mask':y}

    def _preload_data(self):
        with h5py.File(self.h5_file, "r") as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
        return x, y
