from codecs import open
from os import path
from setuptools import find_packages, setup

from pgf_kernel_experiments import __version__

url = 'https://github.com/papamarkou/pgf_kernel_experiments'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pgf_kernel_experiments',
    version=__version__,
    description='PGF kernel experiments',
    long_description=long_description,
    url=url,
    download_url='{0}/archive/v{1}.tar.gz'.format(url, __version__),
    packages=find_packages(),
    license='MIT',
    author='Theodore Papamarkou',
    author_email='theodore.papamarkou@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
    keywords=['compositional kernels', 'Gaussian processes', 'probability generating functions'],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.19.2',
        'torch>=1.11.0',
        'gpytorch>=1.6.0',
        'pgfml>=0.0.1',
        'zarr>=2.13.3',
        'scipy>=1.10.0',
        'geotiff>=0.2.7',
        'gstools>=1.4.1',
        'matplotlib>=3.6.3',
        'cartopy>=0.21.0'
    ]
)
