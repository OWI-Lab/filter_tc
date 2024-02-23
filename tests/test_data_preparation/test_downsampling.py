"""
test_downsampling.py - Module to test the downsampling of filter_tc.data_preparation.downsampling module
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

# global imports
import pytest
import numpy as np

# sdypy_sep005 imports
from sdypy_sep005.sep005 import assert_sep005

# filter_tc imports
from filter_tc.data_preparation.downsampling import Sep005Downsampler

@pytest.fixture
def sample_sep005_data():
    return {
        "data": np.random.rand(1000),  # Random data for testing
        "fs": 1000,  # Original sampling frequency
        "name": "test_measurement",
        "unit_str": "test_unit"
    }

def test_initialization(sample_sep005_data):
    new_fs = 500
    downsampler = Sep005Downsampler(sample_sep005_data, new_fs)
    assert downsampler.new_fs == new_fs

def test_adapt_fs(sample_sep005_data):
    downsampler = Sep005Downsampler(sample_sep005_data, 2)
    adapted_old_fs, adapted_new_fs = downsampler.adapt_fs()
    assert adapted_old_fs == 1000
    assert adapted_new_fs == 2

    downsampler = Sep005Downsampler(sample_sep005_data, 0.5)
    adapted_old_fs, adapted_new_fs = downsampler.adapt_fs()
    assert adapted_old_fs == 2000
    assert adapted_new_fs == 1

def test_check_fs_valid(sample_sep005_data):
    downsampler = Sep005Downsampler(sample_sep005_data, 500)
    downsampler.check_fs()  # Should not raise any exceptions

def test_check_fs_invalid(sample_sep005_data):
    with pytest.raises(ValueError):
        downsampler = Sep005Downsampler(sample_sep005_data, 750)
        downsampler.check_fs()

def test_downsample_array(sample_sep005_data):
    downsampler = Sep005Downsampler(sample_sep005_data, 500)
    downsampled_data = downsampler.downsample_array(sample_sep005_data["data"])
    assert len(downsampled_data) == 500

def test_downsample(sample_sep005_data):
    downsampler = Sep005Downsampler(sample_sep005_data, 500)
    downsampler.downsample()
    downsampled_sep005 = downsampler.downsampled_sep005
    assert len(downsampled_sep005["data"]) == 500
    assert_sep005(downsampled_sep005)  # Ensure the downsampled data still conforms to SEP005 format
