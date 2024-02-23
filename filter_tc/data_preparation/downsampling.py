"""
downsampling.py - Module to downsample SEP005 data to a new sampling frequency.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

from typing import Union, List
import numpy as np
from sdypy_sep005.sep005 import assert_sep005

class Sep005Downsampler:
    def __init__(
        self,
        sep005:Union[dict,List[dict]],
        new_fs: float
        ) -> None:
        """Class to downsample SEP005 data to a new sampling frequency.

        Args:
            sep005 (Union[dict,List[dict]]): Data to be downsampled in the sep005 format.
            new_fs (float): new sampling frequency.
        """
        assert_sep005(sep005)
        self.sep005 = sep005
        self.downsampled_sep005 = {}
        self.new_fs = new_fs
        self.old_fs = sep005["fs"]
        self.check_fs()

    def adapt_fs(self) -> tuple:
        if self.new_fs <= 1:
            adapted_old_fs = self.old_fs / self.new_fs
            adapted_new_fs = 1
        else:
            adapted_old_fs = self.old_fs
            adapted_new_fs = self.new_fs
        return adapted_old_fs, adapted_new_fs

    def check_fs(self) -> None:
        adapted_old_fs, adapted_new_fs = self.adapt_fs()
        if adapted_old_fs % adapted_new_fs != 0:
            raise ValueError('New sampling frequency must be a multiple of the old sampling frequency')
        if adapted_old_fs < adapted_new_fs:
            raise ValueError('New sampling frequency must be smaller than the old sampling frequency')
        
    def downsample_array(self, data:np.ndarray) -> np.ndarray:
        """Function that downsamples the data to the new_fs.

        Args:
            data (np.ndarray): data to be downsampled.

        Returns:
            np.ndarray: downsampled data.
        """
        adapted_old_fs, adapted_new_fs = self.adapt_fs()
        downsampled_data = data[::int(adapted_old_fs/adapted_new_fs)]
        return downsampled_data
    
    def downsample(self) -> Union[dict, List[dict]]:
        """Function that downsamples the measurement data to the new_fs.
        """
        for key in self.sep005.keys():
            if isinstance(self.sep005[key], np.ndarray):
                if len(self.sep005[key]) == len(self.sep005["data"]):
                    self.downsampled_sep005[key] = self.downsample_array(self.sep005[key])
                else:
                    self.downsampled_sep005[key] = self.sep005[key]
            else:
                self.downsampled_sep005[key] = self.sep005[key]
        assert_sep005(self.downsampled_sep005)
    
