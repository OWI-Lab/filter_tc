"""
kalman_filter.py is a module for applying the kalman filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
import datetime
from typing import List, Union, Mapping
import numpy as np

from filter_tc.utils import preprocess_measurements, preprocess_inputs, learn_alpha, define_alpha


SEP005_DEFAULT_FORMAT = '%Y-%m-%d %H:%M:%S%z'


class KalmanFilterBank(list):
    """
    The KalmanFilterBank class manages all kalman filters
    associated with a set of measurements and inputs.
    """
    def __getitem__(
        self,
        item
    ):
        if isinstance(item, slice):
            result = list.__getitem__(self, item)
            return KalmanFilterBank(result)
        elif isinstance(item, int):
            result = list.__getitem__(self, item)
            return result
        elif isinstance(item, str):
            for kf in self:
                if kf.name == item:
                    return kf
            raise ValueError(f'No kalman filter with name "{item}" in {str(self)}')

    @classmethod
    def from_sep005(
        cls,
        measurements:Union[dict, List[dict]],
        q_process_variance: Union[float, np.ndarray],  # Q
        r_measurement_variance: Union[float, np.ndarray],  # R
        a_matrix: np.ndarray = np.array([[1.0]]),
        b_matrix: np.ndarray = np.array([[0.0]]),
        c_matrix: np.ndarray = np.array([[1.0]]),
        inputs:Union[dict, List[dict], None] = None,
        alpha:Union[float, List[float], None] = None,
        posteri_state: np.ndarray = np.array([0]),
        posteri_error_covariance: np.ndarray = np.array([0]),
    ) -> List['KalmanFilter']:
        """Initialize a set of kalman filters from SEP005 compliant measurements.
        FIXME: What if measurements vary greatly and need KalmanFilter with specific settintgs.
        There should be a way to customize the filters, specially the alpha term!
        """
        measurements = preprocess_measurements(measurements)
        if inputs is not None:
            inputs = preprocess_inputs(inputs)
        # In all following code we assume that measurements is a list of 1D measurement,
        # as implemented in the preprocess_measurements function
        kalman_filters = []
        for i, measurement in enumerate(measurements):
            alpha_ = define_alpha(alpha, i, measurement, inputs)


            kalman_filter = KalmanFilter(
                q_process_variance,
                r_measurement_variance,
                posteri_state,
                posteri_error_covariance,
                a_matrix,
                b_matrix,
                c_matrix,
            )
            if 'start_timestamp' in measurement:
                if 'start_timestamp_format' in measurement:
                    timestamp_format = measurement['start_timestamp_format']
                else:
                    # TODO: @Wout, is there a default format for sep005 timestamps?
                    # NOTE: I'm using the default format from the SCB project currently.
                    #sep005_default_format = '%Y-%m-%d %H:%M:%S%z'
                    timestamp_format = SEP005_DEFAULT_FORMAT
                kalman_filter.timestamp = datetime.datetime.strptime(measurement['start_timestamp'], timestamp_format)
            # Add to the list of particle filters
            kalman_filters.append(kalman_filter)
        return cls(kalman_filters)

    @classmethod
    def from_states(
        cls,
        collected_states: dict[str, dict[str, Union[str, float, List[float], np.ndarray]]]
    ) -> List['KalmanFilter']:
        """This function creates a filterbank from a previous filter state
        (e.g. as stored from 'export_states')

        Args:
            collected_states (List[dict]): Collected states of the particle filters to be loaded.
                Settings of the particle filters are stored in a dict.

        Returns:
            List['KalmanFilter']: KalmanFilterBank,
                containing one kalman filter for every measurement.
        """
        kalman_filters = []
        states = {}
        for sensor in collected_states:
            for var in collected_states[sensor]:
                if isinstance(collected_states[sensor][var], list):
                    collected_states[sensor][var] = np.array(collected_states[sensor][var])
            states[sensor] = collected_states[sensor]
            kalman_filter = KalmanFilter(**states[sensor]) # Using the dict as the input for the particle filters
            kalman_filters.append(kalman_filter)
        return cls(kalman_filters)


    def export_states(
        self
    ) -> Mapping[str, Mapping[str, Union[str, float, List[float]]]]:
        """Export the state of every ParticleFilter in the class, e.g. to be stored somewhere

        Returns:
            List[dict]: Collected states of the particle filters to be loaded.
                Settings of the particle filters are stored in a dict.
        """
        collected_states = {}
        for kf in self:
            collected_states[kf.name] = {}
            for var in vars(kf):
                if isinstance(vars(kf)[var], np.ndarray):
                    collected_states[kf.name][var] = vars(kf)[var].tolist()
                elif isinstance(vars(kf)[var], datetime.datetime):
                    collected_states[kf.name][var] = vars(kf)[var].strftime(SEP005_DEFAULT_FORMAT)
                else:
                    collected_states[kf.name][var] = vars(kf)[var]
        return collected_states


    def filter(
        self,
        measurements:Union[dict, List[dict]],
        inputs:Union[dict, List[dict]]
    ) -> List[dict]:
        """Filter all the measurements, 
        using the kalman filters in the partcile filter bank with the specified input.

        Args:
            measurements (Union[dict, List[dict]]): measurements to be filtered.
            inputs (Union[dict, List[dict]]): input to be used for the filtering.

        Returns:
            List[dict]: Filtered measurements.
        """
        measurements = preprocess_measurements(measurements)
        inputs = preprocess_inputs(inputs)
        # Particle filter takes in the temperature and the delta_Temperature
        filter_inputs = np.vstack(
            [
                inputs['data'], #type: ignore
                # Add a zero to the start of the sample to have the same length as the measurements
                np.insert(np.diff(inputs['data']), 0, 0)  #type: ignore
            ]
        )
        filter_outputs = []
        for measurement in measurements:
            kalman_filter = self[measurement['name']]
            if kalman_filter is not None:
                filtered = kalman_filter.filter(
                    measurement['data'],
                    filter_inputs #type: ignore
                )
            else:
                raise ValueError(f'No kalman filter with name "{measurement["name"]}" in {str(self)}')
            filtered_data = measurement.copy()
            filtered_data['data'] = filtered
            filtered_data['name'] = 'filtered_' + measurement['name']
            if kalman_filter.timestamp: # type: ignore
                ## Update the particle filter timestamp to the latest sample
                # TODO: @Wout, is there a default format for sep005 timestamps?
                # NOTE: I'm using the default format from the SCB project currently.
                timestamp_format = SEP005_DEFAULT_FORMAT
                kalman_filter.timestamp =  ( # type: ignore
                    datetime.datetime.strptime(measurement['start_timestamp'], timestamp_format)
                    + datetime.timedelta(seconds=measurement['fs']*len(measurement['data']))
                    )
            filter_outputs.append(filtered_data)
        return filter_outputs


def check_matrix_multiplication(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    """Check if the matrix multiplication is possible.

    Args:
    matrix_a (np.ndarray): The first matrix.
    matrix_b (np.ndarray): The second matrix.

    Raises:
    ValueError: If the matrix multiplication is not possible.
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            f"Matrix multiplication not possible for matrices with shapes {matrix_a.shape} and {matrix_b.shape}"
        )
    

class KalmanFilter:
    """A Kalman filter implementation for data smoothing.
    This kalman filter is designed to be used for temperature compensation.
    """
    def __init__(
            self,
            q_process_variance: Union[float, np.ndarray],  # Q
            r_measurement_variance: Union[float, np.ndarray],  # R
            posteri_state: np.ndarray = np.array([0]),
            posteri_error_covariance: np.ndarray = np.array([0]),
            a_matrix: np.ndarray = np.array([[1.0]]),
            b_matrix: np.ndarray = np.array([[0.0]]),
            c_matrix: np.ndarray = np.array([[1.0]]),
            alpha:Union[float, List[float], None] = None,
            name:Union[str,None] = None,
            timestamp:Union[datetime.datetime,None] = None,
            u_input:Union[np.ndarray,None] = None
    ):
        self.q_process_variance = q_process_variance
        self.r_measurement_variance = r_measurement_variance
        self.posteri_state = posteri_state
        self.posteri_error_covariance = posteri_error_covariance
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.c_matrix = c_matrix
        self.alpha = alpha
        self.name = name
        self.timestamp = timestamp
        self.u_input = u_input

    def input_latest_noisy_measurement(
        self,
        measurement: Union[float, np.ndarray],
        input: np.ndarray = np.array([[0.0]]),
    ) -> Union[float, np.ndarray]:
        """Input the latest noisy measurement and
        update the posteri estimate and posteri error estimate.

        Args:
        measurement (float or np.ndarray): The latest noisy measurement (=y_k).
        """
        # Time Update (Prediction)
        # x_hat_predicted = A * x_hat_posteri + B * u
        check_matrix_multiplication(self.a_matrix, self.posteri_state)
        check_matrix_multiplication(self.b_matrix, input)
        priori_state = self.a_matrix @ self.posteri_state + self.b_matrix @ input #type: ignore
        # P_predicted = A * P_posteri * A.T + Q
        check_matrix_multiplication(self.a_matrix, self.posteri_error_covariance)
        check_matrix_multiplication(self.posteri_error_covariance, self.a_matrix)
        priori_error_covariance = \
            self.a_matrix @ self.posteri_error_covariance @ self.a_matrix.T + self.q_process_variance #type: ignore
        # Measurement Update (Correction)
        # K = P_predicted * H.T * (H * P_predicted * H.T + R).inverse()
        check_matrix_multiplication(self.c_matrix, priori_error_covariance)
        check_matrix_multiplication(priori_error_covariance, self.c_matrix)
        innovation_covariance = \
            self.c_matrix @ priori_error_covariance @ self.c_matrix.T \
            + self.r_measurement_variance
        kalman_gain = \
            priori_error_covariance @ self.c_matrix.T \
            @ np.linalg.inv(innovation_covariance)
        
        # x_hat_posteri = x_hat_predicted + K * (z - H * x_hat_predicted)
        innovation = measurement - self.c_matrix @ priori_state
        self.posteri_state = priori_state + kalman_gain @ innovation

        # P_posteri = (I - K * H) * P_predicted
        self.posteri_error_covariance = \
            (np.eye(self.a_matrix.shape[0]) - kalman_gain @ self.c_matrix) \
            @ priori_error_covariance
        return self.posteri_state
    
    def get_latest_estimated_state(self) -> Union[float, np.ndarray]:
        """
        Get the latest estimated state.

        Returns:
        float or np.ndarray: The latest estimated state.
        """
        return self.posteri_state
    
    def filter(
        self,
        measurements: np.ndarray,
        inputs: np.ndarray,
        ) -> np.ndarray:
        filtered_data = np.zeros(measurements.shape)
        for i, measurement in enumerate(measurements):
            self.input_latest_noisy_measurement(
                measurement,
                inputs[:,i].reshape(-1,1)
            )
            filtered_data[i] = self.get_latest_estimated_state()
        return filtered_data