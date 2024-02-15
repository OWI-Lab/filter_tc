#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Maximillian Weil",
    author_email='maximillian.weil@vub.be',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Filter-based temperature comensation methods to remove the slow temperature trends from strain data with short term events.",
    entry_points={
        'console_scripts': [
            'filter_tc=filter_tc.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='filter_tc',
    name='filter_tc',
    packages=find_packages(include=['filter_tc', 'filter_tc.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/WEILMAX/filter_tc',
    version='0.2.0',
    zip_safe=False,
)
