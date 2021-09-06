from pathlib import PosixPath
from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
# from torch.utils.data import Dataset


class Dataset():
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    """

    def __init__(
        self,
        h5_file: PosixPath,
    ):
        self.h5_file = h5_file
        self.train_index_list = [21,46,59,71,72,81,88,91,104,105,118,136,138,148,149,150,171,175,179,182,184,196,197,
        205,213,214,220,221,222,229,241,242,246,251,255,261,264,271,272,284,287,300,306,319,320,326,327,374,399,
        430,436,450,451,468,472,473,490,499,523,524,525,544,545,552,553,563,568,572,573,589,593,598,599,609,614,
        618,619,635,639,644,645,655,660,664,665,681,685,690,691,701,706,710,711,727,731,741,744,745,748,751,757,760,761,764,
        767,773,776,777,780,783,789,792,793,796,799,805,808,809,812,815,821,824,825,828,831,837,840,841,844,847,853,856,857,860,863]
        self.valid_index_list = [23, 24, 25, 64, 65, 66, 67, 68, 69, 70, 75, 76, 84, 85, 86, 87, 88, 96]

        # Placeholder for catchment attributes stats
        self.df = None

        (self.x, self.y) = self._preload_data()

        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.x[idx][:,1:8,...]
        y = self.y[idx]

        # 进行值到dbz及dbz到掩码的转变
        x = 255 * (x) / 90
        x[x<0] = 0
        x[x>255] = 255
        # x = np.piecewise(x, [x<0, x>=255], [0, 255])

        y = 10*np.log(58.53) + 10*1.6*np.log(y+0.001)
        y = 255 * (y) / 90
        y[y<0] = 0
        y[y>255] = 255
        # y = np.piecewise(y, [y<0, y>=255], [0, 255])



        # convert to torch tensors
        x = torch.from_numpy((x.astype(np.float32))/255)
        # x = torch.from_numpy(x.astype(np.float32))[:,:,...]
        # y = torch.from_numpy(y.astype(np.float32))

        y = torch.from_numpy((y.astype(np.float32))/255)
        print(x.shape, y.shape)
        return {'image':x, 'mask':y}

    def _preload_data(self):

        with h5py.File(self.h5_file, "r") as f:
            if 'train' in self.h5_file:
                # x = f["input_data"][567:570]
                # y = f["target_data"][567:570]
                x = f["input_data"][:]
                y = f["target_data"][:]
                # x = f["input_data"][self.train_index_list]
                # y = f["target_data"][self.train_index_list]
            else:
                x = f["input_data"][:]
                y = f["target_data"][:]
                # x = f["input_data"][self.valid_index_list]
                # y = f["target_data"][self.valid_index_list]

        return x, y
