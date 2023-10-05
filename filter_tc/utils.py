from typing import Union, List
import numpy as np
from sdypy_sep005.sep005 import assert_sep005
from sklearn.linear_model import LinearRegression


def preprocess_measurements(
    measurements:Union[dict, List[dict]]
    ) -> Union[dict, List[dict]]:
    """Simple function that does the initial validation and pre-processing of the measurement data.
    Serves to unite the two flavours of SEP005.

    Args:
        measurements (Union[dict, List[dict]]): measurement data used to be filtered by the particle filter.

    Raises:
        NotImplementedError: Particle filter only implemented for a one dimensional measurement.
        NotImplementedError: Particle filter only implemented for a one dimensional measurement.

    Returns:
        Union[dict, List[dict]]: preprocessed measurement data as a sep_005 compliant dict or List.
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


def preprocess_inputs(
    inputs:Union[dict, List[dict]]
    )-> Union[dict, List[dict]]:
    """Simple function that does the initial validation and pre-processing of the input data.
    Serves to unite the two flavours of SEP005

    Args:
        inputs (Union[dict, List[dict]]): input data used as input to the particle filter.

    Raises:
        NotImplementedError: Particle filter only implemented for a single input.
        NotImplementedError: Particle filter only implemented for a one dimensional input.
        NotImplementedError: Particle filter only implemented for a one dimensional input.

    Returns:
        Union[dict, List[dict]]: preprocessed input data as a sep_005 compliant dict.
    """    
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
    

def assert_pfsettings(pf_settings: dict):
    """Assert the compliance of the particle filter settings.
    # TODO: Decide on the settings and implement this function.

    Args:
        pf_settings (dict): particle filter settings.
    """
    pass



def learn_alpha(
    temperature_input:np.ndarray,
    strain_measurement:np.ndarray
    ) -> float:
    """Function that learns the alpha parameter of the particle filter.
    This is done by fitting a linear regression to the temperature and strain measurement data.

    Args:
        temperature_input (np.ndarray): temperature input data.
        strain_measurement (np.ndarray): strain measurement data.

    Returns:
        float: alpha parameter of the particle filter.
    """
    linregr = LinearRegression()
    linregr.fit(temperature_input, strain_measurement)
    alpha = linregr.coef_[0]
    return alpha
