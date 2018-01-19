
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension
from setuptools.command.install import install

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
    name="dataset",
    version="0.1",
    description="Python package for creating and managing datasets in HDF5 format.",
    long_description="Python package for creating and managing datasets in HDF5 format.",
    author="Yannik Behr and Nial Peters",
    author_email="y.behr@gns.cri.nz",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
    install_requires=['tables', 'numpy'],
    packages=['dataset'],
    test_suite='nose.collector',
    test_require=['nose']
)
