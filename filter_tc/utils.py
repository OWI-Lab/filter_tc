from typing import Union, List

from sdypy_sep005.sep005 import assert_sep005


def preprocess_measurements(measurements:Union[dict, List[dict]]):
    """
    Simple function that does the initial validation and pre-processing of the measurement data.

    Serves to unite the two flavours of SEP005

    Args:
        measurements:

    Returns:

    """
    assert_sep005(measurements)
    # SEP005 allows for some flexibility on how arrays are defined.
    # However I propose that we always consider only 1D arrays (so lists of data). Raise a not implemented error for now
    if isinstance(measurements, list):
        for m in measurements:
            if m['data'].ndim != 1:
                raise NotImplementedError(
                    f'Particle filter functionality not yet implemented for multi-dimensional data arrays')
    else:
        if measurements['data'].ndim != 1:
            raise NotImplementedError(
                f'Particle filter functionality not yet implemented for multi-dimensional data arrays'
            )
        measurements = [measurements]  # Make the dict into a list

    return measurements


def preprocess_inputs(inputs:Union[dict, List[dict]]):

    assert_sep005(inputs)
    # For now we only consider a single input
    if isinstance(inputs, list):
        if len(inputs)>1:
            raise NotImplementedError(
                f'Particle filter functionality not yet implemented for multiple input arrays'
            )

        if inputs[0]['data'].ndim != 1:
            raise NotImplementedError(
                f'Particle filter functionality not yet implemented for multi-dimensional data arrays')

        return inputs[0]

    else:
        if inputs['data'].ndim != 1:
            raise NotImplementedError(
                f'Particle filter functionality not yet implemented for multi-dimensional data arrays')

        return inputs

