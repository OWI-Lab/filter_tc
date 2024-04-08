"""
particle_filter.py is a module for applying the particle filter
as a temperature compensation method.
Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
import datetime
from typing import List, Union, Mapping
import warnings
import scipy as sp  # FIXME: This is a HEAVY dependency, if we can loose it in the future would be nice.
                    # NOTE: It is possible to define a gamma distribution in numpy alternatively (np.random.gamma(shape, scale, size)).
from scipy import stats
import numpy as np
import time
import timeit
import concurrent.futures

from filter_tc.utils import preprocess_measurements, preprocess_inputs, learn_alpha, define_alpha
from filter_tc.custom_distributions import CustomGammaDistribution

SEP005_DEFAULT_FORMAT = '%Y-%m-%d %H:%M:%S%z'

class ParticleFilterBank(list):
    """
    The ParticleFilterBank class manages all particle filters
    associated with a set of measurements and inputs.
    """
    def __getitem__(
        self,
        item
    ):
        if isinstance(item, slice):
            result = list.__getitem__(self, item)
            return ParticleFilterBank(result)
        elif isinstance(item, int):
            result = list.__getitem__(self, item)
            return result
        elif isinstance(item, str):
            for pf in self:
                if pf.name == item:
                    return pf
            raise ValueError(f'No particle filter with name "{item}" in {str(self)}')

    @classmethod
    def from_sep005(
        cls,
        measurements:Union[dict, List[dict]],
        inputs:Union[dict, List[dict], None] = None,
        num_particles:int = 1000,
        r_measurement_noise:float = 0.1,
        q_process_noise:np.ndarray = np.array([0.1, 0.1]),
        loc:float = -0.1,
        alpha:Union[float, List[float], None] = None,
        event_distribution:Union[sp.stats.rv_continuous,None] = None, #type: ignore
    ) -> List['ParticleFilter']:
        """Initialize a set of particle filters from SEP005 compliant measurements.
        FIXME: What if measurements vary greatly and need ParticleFilters with specific settintgs.
        There should be a way to customize the filters, specially the alpha term!

        Args:
            measurements (Union[dict, List[dict]]):All measurements collected
            num_particles (int, optional): Number of particles to use in the filter.
                Defaults to 1000.
            r_measurement_noise (float, optional): The measurement noise. 
                Used to generate the gamma distribution that discribes the events.
                The distribution is used for generating the weights of the particles.
                Defaults to 0.1.
            q_process_noise (np.ndarray, optional): Noise of the process.
                Used to generate the particles.
                Defaults to np.array([0.1, 0.1]).
            scale (float, optional): Scale of the noise added to particles when resampling.
                Defaults to 0.1.
            loc (float, optional): Shift of the gamma distribution,
                discribing the noise on the measurements.
                Defaults to -0.1.
            alpha (Union[float, List[float], None], optional): Thermal expanison coefficient.
                Descibes the relationship between the strain and temperature.
                Defaults to None.

        Returns:
            List['ParticleFilter']: ParticleFilterBank,
                containing one particle filters for every measurement.
        """
        measurements = preprocess_measurements(measurements)
        if inputs is not None:
            inputs = preprocess_inputs(inputs)
        # In all following code we assume that measurements is a list of 1D measurement,
        # as implemented in the preprocess_measurements function
        particle_filters = []
        for i, measurement in enumerate(measurements):
            alpha_ = define_alpha(alpha, i, measurement, inputs)
            particle_filter = ParticleFilter(
                num_particles,
                r_measurement_noise,
                q_process_noise,
                loc,
                alpha_,
                name=measurement['name'],
                event_distribution = event_distribution
            )
            if 'start_timestamp' in measurement:
                if 'start_timestamp_format' in measurement:
                    timestamp_format = measurement['start_timestamp_format']
                else:
                    # TODO: @Wout, is there a default format for sep005 timestamps?
                    # NOTE: I'm using the default format from the SCB project currently.
                    #sep005_default_format = '%Y-%m-%d %H:%M:%S%z'
                    timestamp_format = SEP005_DEFAULT_FORMAT
                particle_filter.timestamp = datetime.datetime.strptime(measurement['start_timestamp'], timestamp_format)
            mean = np.array([measurement['data'][0], 0]) # From initial value of the measurement
            std = np.array([0.1, 0.1])
            particle_filter.create_gaussian_particles(mean, std)
            # Add to the list of particle filters
            particle_filters.append(particle_filter)
        return cls(particle_filters)

    @classmethod
    def from_states(
        cls,
        collected_states: dict[str, dict[str, Union[str, float, List[float], np.ndarray]]]
    ) -> List['ParticleFilter']:
        """This function creates a filterbank from a previous filter state
        (e.g. as stored from 'export_states')

        Args:
            collected_states (List[dict]): Collected states of the particle filters to be loaded.
                Settings of the particle filters are stored in a dict.

        Returns:
            List['ParticleFilter']: ParticleFilterBank,
                containing one particle filters for every measurement.
        """
        particle_filters = []
        states = {}
        for sensor in collected_states:
            for var in collected_states[sensor]:
                if isinstance(collected_states[sensor][var], list):
                    collected_states[sensor][var] = np.array(collected_states[sensor][var])
                if var == 'event_distribution':
                    distribution = \
                        getattr(
                            sp.stats,
                            collected_states[sensor][var]['dist_name']
                            )(
                                collected_states[sensor][var]['dist_args'],
                                **collected_states[sensor][var]['dist_kwds']
                            )
                    collected_states[sensor][var] = distribution

            states[sensor] = collected_states[sensor]

            particle_filter = ParticleFilter(**states[sensor]) # Using the dict as the input for the particle filters
            particle_filters.append(particle_filter)
        return cls(particle_filters)


    def export_states(
        self
    ) -> Mapping[str, Mapping[str, Union[str, float, List[float]]]]:
        """Export the state of every ParticleFilter in the class, e.g. to be stored somewhere

        Returns:
            List[dict]: Collected states of the particle filters to be loaded.
                Settings of the particle filters are stored in a dict.
        """
        collected_states = {}
        for pf in self:
            collected_states[pf.name] = {}
            for var in vars(pf):
                if var == 'event_distribution':
                    collected_states[pf.name][var] = {'dist_name': vars(pf)[var].dist.name, 'dist_args': vars(pf)[var].args, 'dist_kwds': vars(pf)[var].kwds}
                elif isinstance(vars(pf)[var], np.ndarray):
                    collected_states[pf.name][var] = vars(pf)[var].tolist()
                elif isinstance(vars(pf)[var], datetime.datetime):
                    collected_states[pf.name][var] = vars(pf)[var].strftime(SEP005_DEFAULT_FORMAT)
                else:
                    collected_states[pf.name][var] = vars(pf)[var]
        return collected_states


    def filter(
        self,
        measurements:Union[dict, List[dict]],
        inputs:Union[dict, List[dict]]
    ) -> List[dict]:
        """Filter all the measurements, 
        using the particle filters in the partcile filter bank with the specified input.

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
            particle_filter = self[measurement['name']]
            if particle_filter is not None:
                filtered = particle_filter.filter(
                    measurement['data'],
                    filter_inputs #type: ignore
                )
            else:
                raise ValueError(f'No particle filter with name "{measurement["name"]}" in {str(self)}')
            filtered_data = measurement.copy()
            filtered_data['data'] = filtered
            filtered_data['name'] = 'filtered_' + measurement['name']
            if particle_filter.timestamp: # type: ignore
                ## Update the particle filter timestamp to the latest sample
                # TODO: @Wout, is there a default format for sep005 timestamps?
                # NOTE: I'm using the default format from the SCB project currently.
                timestamp_format = SEP005_DEFAULT_FORMAT
                particle_filter.timestamp =  ( # type: ignore
                    datetime.datetime.strptime(measurement['start_timestamp'], timestamp_format)
                    + datetime.timedelta(seconds=measurement['fs']*len(measurement['data']))
                    )
            filter_outputs.append(filtered_data)
        return filter_outputs


class ParticleFilter:
    """A Particle filter implementation for data smoothing.
    This particle filter is designed to be used for temperature compensation.

    Attributes:
        num_particles (int, optional): Number of particles to use in the filter.
            Defaults to 1000.
        r_measurement_noise (float, optional): The measurement noise. 
            Used to generate the gamma distribution that discribes the events.
            The distribution is used for generating the weights of the particles.
            Defaults to 0.1.
        q_process_noise (np.ndarray, optional): Noise of the process.
                Used to generate the particles.
                Defaults to np.array([0.1, 0.1]).
        scale (float, optional): Scale of the noise added to particles when resampling.
            Defaults to 0.1.
        loc (float, optional): Shift of the gamma distribution,
            discribing the noise on the measurements.
            Defaults to -0.1.
        alpha (Union[float, List[float], None], optional): Thermal expanison coefficient.
            Descibes the relationship between the strain and temperature.
            Defaults to None.
        particles (Union[np.ndarray,None], optional): Last generated particles of the particle filter.
            Defaults to None.
        weights (Union[np.ndarray,None], optional): Last generated weights for weighting the particles.
            Defaults to None.
        event_distribution (Union[sp.stats.rv_continuous,None], optional): Distribution of the events.
            Defaults to None.
        name (Union[str,None], optional): Name of the filtered measurements.
            Defaults to None.
        timestamp (Union[datetime.datetime,None], optional): Timestamp of the final state of the particle filter.
            Used to know if the particle filter can be appended with new measurements.
            Defaults to None.
        u_input (Union[np.ndarray,None], optional): Last input of the particle filter.
            Defaults to None.
    """
    def __init__(
            self,
            num_particles:int = 1000,
            r_measurement_noise:float = 0.1, 
            q_process_noise:np.ndarray = np.array([0.1, 0.1]),
            loc:float = -0.1,
            alpha:Union[float, List[float], None] = None,
            particles:np.ndarray = np.array([]),
            weights:np.ndarray = np.array([]),
            event_distribution:Union[sp.stats.rv_continuous,None] = None, #type: ignore
            name:Union[str,None] = None,
            timestamp:Union[datetime.datetime,None] = None,
            u_input:Union[np.ndarray,None] = None
    ):
        self.num_particles = num_particles
        self.r_measurement_noise = r_measurement_noise
        self.q_process_noise = q_process_noise if q_process_noise is not None else np.array([0.1, 0.1])
        # Properties defining the event distribution
        self.loc = loc
        self.event_distribution = event_distribution
        if self.event_distribution is None:
            # TODO: Replace scipy by numpy to define the distribution
            self.event_distribution = \
                sp.stats.invweibull( #type: ignore
                    1,
                    scale=self.r_measurement_noise,
                    loc=self.loc
                )
        # NOTE: Alternative numpy distribution (doesn't work yet)
        # self.np_event_distribution = \
        #    CustomGammaDistribution(1 - self.loc / self.r_measurement_noise, self.r_measurement_noise, self.loc)
        self.alpha = alpha
        # Properties defining the last observed state of the particle filter
        self.particles = particles
        if len(self.particles) == 0:
            self.particles = np.zeros(self.num_particles)
        self.weights = weights
        if len(self.weights) == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        self.u_input = u_input       
        # Additional properties for administration
        self.name = name
        self.timestamp = timestamp
        
    def create_gaussian_particles(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """Create a set of particles with a normal distribution around the mean.
        
        Args:
            mean (np.ndarray): mean of the generated particles.
            std (np.ndarray): standard deviation of the generated particles.
        """ 
        self.particles = np.empty(self.num_particles)
        self.particles = \
            mean[0] + (np.random.randn(self.num_particles) * std[0])

    def predict(
        self,
        u_input: np.ndarray
    ) -> None:
        """ Randomly generate a bunch of particles and
        Move according to control input u (measured temperature)
        with input noise q (std measured temperature).
        q controls the spread of the generated particles.
        
        Args:
            u_input (np.ndarray): input of the particle filter.
                The input is a 2D array with the first column being the temperature
                and the second column being the delta_temperature.
        """
        # update Ta
        self.u_input = u_input
        self.alpha = define_alpha(self.alpha)
        if self.particles is not None:
            self.particles += \
                u_input[1] * self.alpha \
                + (np.random.randn(self.num_particles) * self.q_process_noise[0])
        else:
            raise ValueError('No particles found, please initialize the particles first.')

    def update(
        self,
        y_measurement: float,
        loading:str = 'tension'
    ) -> None:
        """ Incorporate measurement y (measured strain)
        with measurement noise r."""
        # compute likelihood of measurement
        if loading == 'tension':
            distance = y_measurement - self.particles
        elif loading == 'compression':
            distance = self.particles - y_measurement
        else:
            raise ValueError('Loading must be either "tension" or "compression"')
        if self.event_distribution is None:
            raise ValueError('No event distribution found, please initialize the event distribution first.')
        else:
            self.weights *= \
                self.event_distribution.pdf(distance) #self.np_event_distribution.pdf(distance)
        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    def estimate(
        self
    ):
        """returns mean and variance of the weighted particles"""
        pos = self.particles
        mean = np.average(pos, weights=self.weights, axis=0)
        # var = np.average((pos - mean) ** 2, weights=self.weights, axis=0)
        return mean #, var

    def simple_resample(
        self
    ):
        """Discard highly improbable particle and replace them with copies of the more probable particles.
        Resample particles with replacement according to weights.

        """
        cumulative_sum = np.cumsum(self.weights) #type: ignore
        # normalize the cumulative sum to be in [0, 1]
        cumulative_sum /= cumulative_sum[self.num_particles-1]
        randoms = np.random.rand(self.num_particles)

        # Choose the particle indices based on the cumulative sum
        indexes = np.searchsorted(cumulative_sum, randoms)
        self.particles[:] = self.particles[indexes]

        # keep the weights of the resampled particles
        self.weights[:] = self.weights[indexes]
        # normalize the weights
        self.weights /= np.sum(self.weights) #type: ignore

    def filter(
        self,
        measurements: np.ndarray,
        input: np.ndarray,
        loading: str = 'tension'
    ) -> np.ndarray:
        """Filter the data using the particle filter.

        Args:
            measurements (np.ndarray): measurements to be filtered.
            input (np.ndarray, optional): input to be used for the filtering.
                Defaults to np.array([]).
            loading (str, optional): Loading type, has to be tension or comperssion.
                Defaults to 'tension'.

        Returns:
            np.ndarray: Filtered measurements.
        """
        if self.alpha is None:
            self.alpha = learn_alpha(input[:,i], measurements)
        if loading not in ['tension', 'compression']:
            raise ValueError('Loading must be either "tension" or "compression"')
        predictions = np.zeros(len(measurements))
        for i, measurement in enumerate(measurements):
            self.predict(input[:,i])
            self.update(measurement, loading=loading)
            self.simple_resample()
            prediction = self.estimate()
            predictions[i] = prediction
        return predictions
