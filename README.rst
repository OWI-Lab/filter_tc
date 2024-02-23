=============================================
Particle Filter for Temperature Compensation
=============================================

This Python package provides filter-based temperature compensation methods to remove the slow temperature trends from strain data with short-term events.
An implementation of a particle filter is given, specifically designed for temperature compensation in various measurement scenarios.

Features
--------

* Particle filter implementation for temperature compensation.
* Management of multiple particle filters with `ParticleFilterBank`.
* Support for SEP005 compliant measurements.
* Customizable particle filter settings including noise levels and thermal expansion coefficients.

Installation
------------

Ensure you have Python 3.9 or more recent installed on your system. You can install this package using the following command:

.. code-block:: bash

    $ pip install git+https://github.com/WEILMAX/filter_tc.git

Usage
-----

* Example for using a single ParticleFilter for temperature compensation:
.. code-block:: python

        from filter_tc.particle_filter import ParticleFilter

        # Create a particle filter
        measurements = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Replace with actual measurements
        inputs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Replace with actual inputs (temperature data)

        num_particles=1000

        pf = ParticleFilter(
        num_particles=num_particles,
        r_measurement_noise=5e1,
        q_process_noise=np.array([2, 1]),
        scale=5e-3,
        loc=-10)

        # Run the particle filter
        pf.filter(measurements, inputs, loading='tension')
..

* Example for using a ParticleFilterBank for managing multiple ParticleFilters:
.. code-block:: python

        from filter_tc.particle_filter import ParticleFilterBank

        # Create a particle filter bank
        measurements = [
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # Replace with actual measurements
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Replace with actual measurements
        ]
        inputs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Replace with actual inputs (temperature data)

        num_particles=1000

        pfb = ParticleFilterBank(
        num_particles=num_particles,
        r_measurement_noise=5e1,
        q_process_noise=np.array([2, 1]),
        scale=5e-3,
        loc=-10)

        # Run the particle filter
        filtered = pfb.filter(measurements, inputs, loading='tension')

        # Export the particle filter states to late use
        pfb_states = pfb.export_states()

        pfb2 = ParticleFilterBank().from_states(pfb_states)
..

Resulting filtering
-------------------

Examples for applying the Particle filter for tension and compression loading measurements are given in the notebooks.
The following figures show the concept of the Particle Filter and the resulting data after removing the PF output from the measurements to approximate events.

* Example of inputs, measurements, particle propagations (grey) and Particle Filter output:

  .. figure:: figures/particle_filter/particle_filter_concept.png
     :align: center
     :alt: Example of applying a PF to measurements with given inputs.

* Example of the data after removing the PF output from the measurements to approximate events:

  .. figure:: figures/particle_filter/filtered_data.png
     :align: center
     :alt: Example of temperature compensated data by a PF.

Credits
-------

The implementation is based on the concepts from `Kalman and Bayesian Filters in Python <https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb>`_ and adapted by Maximillian Weil for specialized use.

This package was created in the context of the NWE Interreg 'Smart Circular Bridge' project.
For more information, see https://vb.nweurope.eu/projects/project-search/smart-circular-bridge-scb-for-pedestrians-and-cyclists-in-a-circular-built-environment/