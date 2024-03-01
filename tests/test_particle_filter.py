# PyTests for tc
import numpy as np
import pytest
from sdypy_sep005.sep005 import assert_sep005

from filter_tc.particle_filter import ParticleFilterBank

def test_from_single_sep005():
    no_samples = 1000
    lf_frequency = 2
    lf_amplitude = 200
    t = np.linspace(0, 1, no_samples)  # Timevector
    y_lf = lf_amplitude * np.sin(
        2 * np.pi * t * lf_frequency + np.pi / 8
    )

    input = {
        'data': y_lf,
        'name': 'test_lf',
        'unit_str': '°C',
        'fs': 1
    }

    measurements = {
        'data': y_lf,
        'name': 'test_measurement',
        'unit_str': 'microstrain',
        'fs': 1
    }

    pfb = ParticleFilterBank.from_sep005(measurements)

    assert isinstance(pfb, ParticleFilterBank)
    assert len(pfb) == 1

    filtered = pfb.filter(measurements, input)
    # Output of the function should be SEP005
    assert_sep005(filtered)
    assert filtered[0]['name'] == 'filtered_test_measurement'

def test_not_inplementederrors():
    """
    Not all variants of SEP005 can be handled assert a Not implemented Error

    Returns:

    """
    measurements = {
        'data': np.ones([3,100]),
        'name': 'test_measurement',
        'unit_str': 'microstrain',
        'fs': 1
    }
    assert_sep005(measurements)

    with pytest.raises(NotImplementedError):
        _ = ParticleFilterBank.from_sep005(measurements)

import pytest
import numpy as np
from filter_tc.particle_filter import ParticleFilterBank, ParticleFilter

@pytest.fixture
def example_particle_filter():
    return ParticleFilter()

def test_create_gaussian_particles(example_particle_filter):
    # Test the create_gaussian_particles method of ParticleFilter class
    # Setup
    mean = np.array([25, 0])  # Example mean
    std = np.array([0.1, 0.1])  # Example standard deviation

    # Exercise
    example_particle_filter.create_gaussian_particles(mean, std)

    # Verify
    assert example_particle_filter.particles.shape == (example_particle_filter.num_particles, 2)
    # Check if particles are generated around the mean
    assert np.allclose(np.mean(example_particle_filter.particles, axis=0), mean, atol=1e-1)

# More tests for other methods...

@pytest.fixture
def example_particle_filter_bank():
    measurements = [
        {'name': 'measurement1', 'data': np.array([1, 2, 3]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'},
        {'name': 'measurement2', 'data': np.array([4, 5, 6]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'}
    ]
    inputs = [{'name': 'input', 'data': np.array([1, 2, 3]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'}]  # Example input
    num_particles = 1000
    r_measurement_noise = 0.1
    q_process_noise = np.array([0.1, 0.1])
    scale = 0.1
    loc = -0.1
    alpha = None
    event_distribution = None

    return ParticleFilterBank.from_sep005(measurements, inputs, num_particles, r_measurement_noise,
                                            q_process_noise, scale, loc, alpha, event_distribution)

def test_from_sep005(example_particle_filter_bank):
    # Test the from_sep005 method of ParticleFilterBank class
    assert len(example_particle_filter_bank) == 2
    measurements = [
        {'name': 'measurement1', 'data': np.array([1, 2, 3]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'},
        {'name': 'measurement2', 'data': np.array([4, 5, 6]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'}
    ]
    inputs = [{'name': 'input', 'data': np.array([1, 2, 3]), 'start_timestamp': '2024-03-01 12:00:00+00:00', 'fs': 1, 'unit_str': 'test_untit'}]  # Example input
    num_particles = 1000
    r_measurement_noise = 0.1
    q_process_noise = np.array([0.1, 0.1])
    for particle_filter, measurement in zip(example_particle_filter_bank, measurements):
        assert particle_filter.num_particles == num_particles
        assert particle_filter.r_measurement_noise == r_measurement_noise
        # Check other attributes...

# More tests for other methods...
