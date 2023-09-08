"""Apply the particle filter to an sdypy-sep005 data channel
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from filter_tc.sdypy_sep005_pandas import *
from filter_tc.particle_filter import *
from sdypy_sep005.sep005 import assert_sep005

## FIXME: Need a sep-005 typehint
def temp_comp_pf(measurements: dict,
                 inputs: dict, loading: str,
                 num_particles: int,
                 r_measurement_noise: float,
                 q_process_noise: np.ndarray,
                 scale: float
                 ):
    from filter_tc.particle_filter import ParticleFilter
    assert_sep005(measurements)
    assert_sep005(inputs)
    if np.shape(inputs['data'])[1] != np.shape(measurements['data'])[1]:
        raise ValueError("Input data and all measurements must have the same number of data points")
    if np.shape(inputs['data'])[0] != 1:
        raise ValueError("Input data must be a single row")
    ## FIXME: Create a pf settings assert function
    #assert_pfsettings(pf_settings)
    filtered_data = measurements.copy()
    filtered_data['data'] = []
    filtered_data['name'] = 'filtered_data_' + measurements['name']
    filtered_data['temperature_trend_data'] = []

    filtered_data['filters'] = []
    # loop over all the channels in the data
    for i, sensor_data in enumerate(measurements['data']):
        # Initialyse a particle filter for every channel
        pf = ParticleFilter(
                num_particles,
                r_measurement_noise ,
                q_process_noise,
                scale)
        # Store the pf with settings in the sep-005 data
        filtered_data['filters'].append(pf)
        # Create a filter inputs for the filter,
        # this consists of the temperature data and the differential.
        filter_inputs = np.vstack([
            inputs['data'][0],
            np.concatenate([np.array([0]), np.diff(inputs['data'][0])])])
        # Apply the filter on the sensor_data
        filter_output = pf.filter(
            sensor_data,
            filter_inputs,
            loading)
        # Add filter output to the temperature_trend_data
        filtered_data['temperature_trend_data'].append(filter_output.tolist())
        #np.vstack([temperature_trend_data['data'], filter_output])
        # Add filtered data to the filtered_data
        filtered_data_ = sensor_data - filter_output
        #np.vstack([filtered_data['data'], filtered_data_])
        filtered_data['data'].append(filtered_data_.tolist())
    
    filtered_data['data'] = np.array(filtered_data['data'])
    filtered_data['temperature_trend_data'] = np.array(filtered_data['temperature_trend_data'])
    
    # Ensure the outputs are sdypy sep-005 compliant
    # FIXME:: REQUIRED ensure that the columns remain the same except for data with measurements
    assert_sep005(filtered_data)
    return filtered_data
        