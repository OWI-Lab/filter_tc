"""
band_quality_control.py - Module to control quality of SEP005 data through an acceptance band.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

from typing import Union, List
import pandas as pd
import numpy as np
from sdypy_sep005.sep005 import assert_sep005
from filter_tc.utils import sep005_get_timestamps


class Sep005BandQualityControl:
    def __init__(
        self,
        sep005: Union[dict,List[dict]],
        lower: float,
        upper: float
        ) -> None:
        """Class to apply band quality control to SEP005 data.

        Args:
            sep005 (Union[dict,List[dict]]): Data to be checked in the sep005 format.
            lower (float): lower accepted limit of the band.
            upper (float): upper accepted limit of the band.
        """
        assert_sep005(sep005)
        self.sep005 = sep005
        self.lower = lower
        self.upper = upper
        self.quality_controlled_sep005 = self.sep005.copy()

    def time_imputation(self, start_timestamp_format: str = "%Y-%m-%dT%H:%M:%S%z"):
        """Imputes missing time values in the SEP005 data.

        This method checks if the time array is present in the SEP005 data. If not, it
        generates a time array based on the sampling frequency and the number of data
        points. This is useful for datasets where the time array is not explicitly
        provided.

        Args:
            start_timestamp_format (str, optional): Format of the start timestamp. Defaults to "%Y-%m-%d %H:%M:%S%z".
        """
        if "time" not in self.quality_controlled_sep005.keys():
            if "start_timestamp" not in self.quality_controlled_sep005.keys():
                raise ValueError("No time or start_timestamp in sep005")
            else:
                if "start_timestamp_format" in self.quality_controlled_sep005.keys():
                    start_timestamp_format = self.quality_controlled_sep005["start_timestamp_format"]
                self.quality_controlled_sep005["time"] = sep005_get_timestamps(self.quality_controlled_sep005, start_timestamp_format)
        assert_sep005(self.quality_controlled_sep005)
    
    def out_of_bound_detection(self, **kwargs):
        """Detects data points outside the specified band limits and marks them as NaN.

        This method identifies the indices where data points in `self.sep005['data']`
        are either below the `lower` limit or above the `upper` limit and replaces
        these out-of-bound values with NaN.
        """
        self.time_imputation(**kwargs)
        out_of_bound_indices = np.where((self.quality_controlled_sep005["data"] < self.lower) | (self.quality_controlled_sep005["data"] > self.upper))
        self.quality_controlled_sep005["data"][out_of_bound_indices] = np.nan
        assert_sep005(self.quality_controlled_sep005)

    def out_of_bound_interpolation(self, **kwargs):
        """Interpolates NaN values in the data array.

        First calls `out_of_bound_detection` to mark out-of-bound data points as NaN.
        Then, it interpolates these NaN values based on neighboring data points, providing
        a continuous data set without abrupt gaps or spikes.
        """     
        self.out_of_bound_detection(**kwargs)
        nan_indices = np.where(np.isnan(self.quality_controlled_sep005["data"]))
        try:
            self.quality_controlled_sep005["data"][nan_indices] = \
                np.interp(
                    nan_indices,
                    np.where(~np.isnan(self.quality_controlled_sep005["data"]))[0],
                    self.quality_controlled_sep005["data"][~np.isnan(self.quality_controlled_sep005["data"])])
        except:
            print('forward fill')
            self.quality_controlled_sep005["data"] = pd.Series(self.quality_controlled_sep005["data"]).fillna(method='ffill').values
        assert_sep005(self.quality_controlled_sep005)
        
    def out_of_bound_removal(self, **kwargs):
        """Removes data points that are outside the specified band limits.

        After identifying out-of-bound data points via `out_of_bound_detection`,
        this method removes the corresponding entries from both the `data` and
        `time` arrays in `self.quality_controlled_sep005`, resulting in a dataset
        with only the values within the specified band limits.
        """     
        self.out_of_bound_detection(**kwargs)
        # remove the positions in the time array where the data is nan for any of the columns
        self.quality_controlled_sep005["time"] = self.quality_controlled_sep005["time"][
            np.where(~np.isnan(self.quality_controlled_sep005["data"]))[0]
        ]
        self.quality_controlled_sep005["data"] = self.quality_controlled_sep005["data"][
            np.where(~np.isnan(self.quality_controlled_sep005["data"]))[0]
        ]
        assert_sep005(self.quality_controlled_sep005)