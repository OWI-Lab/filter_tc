"""Apply the particle filter to an sdypy-sep005 data channel
"""

import numpy as np
from filter_tc.particle_filter import ParticleFilter
from sdypy_sep005.sep005 import assert_sep005


## FIXME: Need a sep-005 typehint
def temp_comp_pf(measurements: dict,
                 inputs: dict,
                 loading: str,
                 num_particles: int,
                 r_measurement_noise: float,
                 q_process_noise: np.ndarray,
                 scale: float
                 ):
    """
    Process

    NOTE: all input data should be presented in a SEP005 compliant

    Args:
        measurements:
        inputs:
        loading:
        num_particles:
        r_measurement_noise:
        q_process_noise:
        scale:

    Returns:

    """
    # For now I'm going to assume that the measurements are also always a single channel, higher dimensionality
    # should be resolved at a higher level, as there are two options there.

    # Asserting both the inputs and measurements are SEP005 compliant (TODO: should be caught at a higher level)
    assert_sep005(measurements)
    assert_sep005(inputs)

    # Check compability
    if len(inputs['data']) != len(measurements['data']):
        raise ValueError(
            f"Input data and all measurements must have the same number of data points,"
            f" now ({len(inputs['data'])},{len(measurements['data'])})"
        )


    ## FIXME: Create a pf settings assert function
    # assert_pfsettings(pf_settings)

    filtered_data = measurements.copy()
    filtered_data['data'] = []
    filtered_data['name'] = 'filtered_data_' + measurements['name']
    filtered_data['temperature_trend_data'] = []

    filtered_data['filters'] = []

    sensor_data = measurements['data']
    # Initialize a particle filter for every channel
    pf = ParticleFilter(
        num_particles,
        r_measurement_noise,
        q_process_noise,
        scale)
    # Store the pf with settings in the sep-005 data
    filtered_data['filters'].append(pf)
    # Create a filter inputs for the filter,
    # this consists of the temperature data and the differential.
    filter_inputs = np.vstack(
        [
            inputs['data'],
            np.insert(np.diff(inputs['data']), 0, 0) # Add a zero to the start of the sample
        ]
    )
    # Apply the filter on the sensor_data
    filter_output = pf.filter(
        sensor_data,
        filter_inputs,
        loading
    )
    # Add filter output to the temperature_trend_data
    filtered_data['temperature_trend_data'].append(filter_output.tolist())
    # np.vstack([temperature_trend_data['data'], filter_output])
    # Add filtered data to the filtered_data
    filtered_data_ = sensor_data - filter_output
    # np.vstack([filtered_data['data'], filtered_data_])
    filtered_data['data'].append(filtered_data_.tolist())

    filtered_data['data'] = np.array(filtered_data['data'])
    filtered_data['temperature_trend_data'] = np.array(filtered_data['temperature_trend_data'])

    return filtered_data
