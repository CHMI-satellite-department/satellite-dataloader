#!/usr/bin/env python3

import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='satellite-dataloader',
    version='0.0.1',
    description='Satellite dataloader',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jakub Seidl',
    author_email='jakub.seidl@email.cz',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    packages=['satdl'],
    install_requires=['numpy>=0.19', 'pyproj==3.2', 'rasterio==1.2.6', 'satpy>=0.29', 'trollsift', 'xarray>=0.19'],
    scripts=[]
)
