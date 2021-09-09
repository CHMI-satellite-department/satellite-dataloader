#!/usr/bin/env python3

import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(name='meteotinder-dataloader',
                 version='0.0.1',
                 description='Satelite data dataloader',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 author='Jakub Seidl',
                 author_email='jakub.seidl@email.cz',
                 classifiers=[
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8'
                 ],
                 packages=['mtdl'],
                 install_requires=['numpy', 'pyproj', 'rasterio', 'satpy', 'trollsift', 'xarray'],
                 scripts=[]
                 )


