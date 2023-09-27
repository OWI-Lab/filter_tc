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

def test_filter_performance():
    assert True
