"""
This module contains functions to generate test data for the sensitivity analysis.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

import numpy as np
from scipy.signal import gausspulse

def generate_trend(
    trend: dict[str, float] = {
        'amplitude': 200.0,
        'frequency': 2.0,
        'cycles':2.0,
        'phase': 0.0
        },
    no_samples:int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, no_samples)   # Timevector

    y_lf = trend['amplitude']/2*np.sin(
        trend['cycles']*np.pi*t*trend['frequency'] + trend['phase']
    )
    return t, y_lf

def generate_events(
    events: dict[str, list] = {
        'amplitude': [100],
        'duration': [100],
        'occurence': [0.5]
        },
    no_samples:int = 1000
    ) -> np.ndarray:
    t = np.linspace(0, 1, no_samples)
    y_events_list = []

    for i in range(len(events['amplitude'])):
        _ , y_event = gausspulse(t - events['occurence'][i], fc=(2*no_samples*np.pi)/(events['duration'][i]), retenv=True) # type: ignore
        y_event = y_event * events['amplitude'][i]
        y_events_list.append(y_event)

    y_events = np.sum(y_events_list, axis=0)
    return y_events


def generate_test_data(
    trend: dict[str, float] = {
        'amplitude': 200.0,
        'frequency': 2.0,
        'cycles':2.0,
        'phase': 0.0
        },
    events: dict[str, list] = {
        'amplitude': [100],
        'duration': [100],
        'occurence': [0.5]
        },
    no_samples:int = 1000,
    noise_amplitude:float = 4.0,
    loading: str = 'compression',
    amplitude_shift: float = 1.0,
    time_shift: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Generate trend
    t, y_lf = generate_trend(trend, no_samples)
    # Generate events
    y_events = generate_events(events, no_samples)
    # Add noise to trend y_lf data
    y_noise = np.random.normal(0, noise_amplitude, no_samples)
    # Combine trend, events and noise depending on the loading type
    if loading == 'compression':
        y_measurements = y_lf + y_events + y_noise
    elif loading == 'tension':
        y_measurements = y_lf - y_events + y_noise
    else:
        raise ValueError('Loading must be either "compression" or "tension"')
    
    # Apply amplitude shift
    input_ = y_lf * amplitude_shift
    #Apply phase shift
    input_samples_shift = int(time_shift*no_samples)
    input_ = np.append(input_[input_samples_shift:], input_[:input_samples_shift])

    return t, y_measurements, input_, y_lf, y_events, y_noise


def create_inputs(
        event_to_trend_amplitude = 1/2,
        event_length_to_trend_period = 1/2,
        noise_to_trend = 1/50,
        no_samples = 1000,
        trend_freq = 4.0,
        trend_amplitude = 200
        ):
        noise_amplitude = trend_amplitude*noise_to_trend
        occurences = [(1/4)*(1/trend_freq), (1/trend_freq), (2/trend_freq)+(3/4)*(1/trend_freq), (3/trend_freq)+(1/2)*(1/trend_freq)]
        events_amplitude = [trend_amplitude*event_to_trend_amplitude]*len(occurences)
        event_length = [no_samples*event_length_to_trend_period/trend_freq]*len(occurences)
        trend = {
        'amplitude': trend_amplitude,
        'frequency': trend_freq,
        'cycles':2.0,
        'phase': 0.0
        }
        events = {
                'amplitude': events_amplitude,
                'duration': event_length,
                'occurence': occurences
                }
        return trend, events, noise_amplitude, no_samples


def create_datasets(
    event_to_trend_amplitudes,
    event_length_to_trend_periods,
    time_shifts,
    amplitude_shift,
    loading,
    noise_to_trend = 1/50,
    no_samples = 1000,
    trend_freq = 4.0,
    trend_amplitude = 200
    ) -> dict:

    datasets = {}

    for event_to_trend_amplitude in event_to_trend_amplitudes:
        for event_length_to_trend_period in event_length_to_trend_periods:
            for time_shift in time_shifts:
                trend, events, noise_amplitude, no_samples = \
                    create_inputs(
                        event_to_trend_amplitude,
                        event_length_to_trend_period,
                        noise_to_trend = noise_to_trend,
                        no_samples = no_samples,
                        trend_freq = trend_freq,
                        trend_amplitude = trend_amplitude
                    )
                t, y_measurements, input_, y_lf, y_events, y_noise = \
                    generate_test_data(
                        trend,
                        events,
                        no_samples,
                        noise_amplitude,
                        loading,
                        amplitude_shift,
                        time_shift
                    )
                datasets[(event_to_trend_amplitude, event_length_to_trend_period, time_shift)] = {
                    't':t,
                    'y_measurements':y_measurements,
                    'input_':input_,
                    'y_lf':y_lf,
                    'y_events':y_events,
                    'y_noise':y_noise
                }
    return datasets