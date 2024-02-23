"""
test_band_quality_control.py - Module to test the quality control of filter_tc.data_preparation.band_quality_control module
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

import pytest
import numpy as np
from filter_tc.data_preparation.band_quality_control import Sep005BandQualityControl
from sdypy_sep005.sep005 import assert_sep005
from filter_tc.utils import sep005_get_timestamps

@pytest.fixture
def sample_sep005_data():
    return {
        "data": np.random.rand(100),  # Random data for testing
        "start_timestamp": "2023-01-01T00:00:00+00:00",  # Example start timestamp
        "fs": 100,  # Original sampling frequency
        "name": "test_measurement",
        "unit_str": "test_unit"
    }

def test_initialization(sample_sep005_data):
    lower = 0.2
    upper = 0.8
    qc = Sep005BandQualityControl(sample_sep005_data, lower, upper)
    assert qc.lower == lower
    assert qc.upper == upper

def test_time_imputation(sample_sep005_data):
    qc = Sep005BandQualityControl(sample_sep005_data, 0.2, 0.8)
    qc.time_imputation()
    assert "time" in qc.quality_controlled_sep005

def test_out_of_bound_detection(sample_sep005_data):
    qc = Sep005BandQualityControl(sample_sep005_data, 0.2, 0.8)
    qc.out_of_bound_detection()
    assert np.any(np.isnan(qc.quality_controlled_sep005["data"]))

def test_out_of_bound_interpolation(sample_sep005_data):
    qc = Sep005BandQualityControl(sample_sep005_data, 0.2, 0.8)
    qc.out_of_bound_interpolation()
    assert not np.any(np.isnan(qc.quality_controlled_sep005["data"]))

def test_out_of_bound_removal(sample_sep005_data):
    qc = Sep005BandQualityControl(sample_sep005_data, 0.2, 0.8)
    qc.out_of_bound_removal()
    assert not np.any(np.isnan(qc.quality_controlled_sep005["data"]))
    assert len(qc.quality_controlled_sep005["data"]) == len(qc.quality_controlled_sep005["time"])

def test_time_imputation_error(sample_sep005_data):
    del sample_sep005_data["start_timestamp"]  # Simulate missing timestamp
    qc = Sep005BandQualityControl(sample_sep005_data, 0.2, 0.8)
    with pytest.raises(ValueError):
        qc.time_imputation()