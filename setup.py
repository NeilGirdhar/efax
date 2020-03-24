#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='efax',
    version='0.2',
    description=(
        'Exponential families for JAX'),
    author='Neil Girdhar',
    author_email='mistersheik@gmail.com',
    project_urls={
        "Bug Tracker": "https://github.com/NeilGirdhar/efax/issues",
        "Source Code": "https://github.com/NeilGirdhar/efax",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=[],
    python_requires='>=3.7',
    install_requires=['ipromise>=1.5',
                      'jax>=0.1.61',
                      'jaxlib>=0.1.42',
                      'nptyping>=0.3.1',
                      'numpy>=1.18.2',
                      'scipy>=1.4.1'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
)
