from typing import Union, List
import numpy as np
import warnings
import datetime
from sdypy_sep005.sep005 import assert_sep005
from sklearn.linear_model import LinearRegression

def sep005_get_timestamps(sep005: dict, start_timestamp_format="%Y-%m-%d %H:%M:%S%z"):
    assert_sep005(sep005)
    if "start_timestamp" in sep005:
        if "start_timestamp_format" in sep005:
            start_timestamp_format = sep005["start_timestamp_format"]
        start_timestamp = datetime.datetime.strptime(sep005["start_timestamp"], start_timestamp_format)
        interval = datetime.timedelta(seconds=1/sep005["fs"])
        timestamps = start_timestamp + interval*np.arange(len(sep005["data"]))
        return timestamps
    else:
        print("No start_timestamp in sep005")


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


def define_alpha(
    alpha:Union[float, List[float], None] = None,
    i:int = 0,
    measurement:Union[dict, List[dict], None] = None,
    inputs:Union[dict, List[dict], None] = None
) -> float:
    """Function that defines the alpha parameter of the particle filter.

    Args:
        alpha (Union[float, List[float], None], optional): Relationship term between input and measurements.
            Can be given as a single float for all measurements,
            a list of one floats for every measurement,
            or None to learn alpha from the data if the input is given.
            If nothing is given, it is set to 1.0.
            Defaults to None.

    Returns:
        float: alpha parameter of the particle filter for the given measurement.
    """
    if isinstance(alpha, float): # Constant alpha for all measurements
        alpha_ = alpha
    elif isinstance(alpha, list): # Specific alpha for every measurement
        alpha_ = alpha[i]
        i += 1
    elif alpha is None and inputs is not None: # Learn alpha from the data if no alpha is given
        alpha_ = learn_alpha(inputs['data'].reshape(-1, 1), measurement['data']) #type: ignore
    else: # Constant alpha for all measurements
        alpha_ = 1.0
        warnings.warn('No alpha and no input given, using default value of 1.0')
    return alpha_


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
