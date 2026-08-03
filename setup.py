from setuptools import setup, find_packages

setup(
    name='wormtrails',
    version='1.6.0',
    author='Christopher Dante Ashih',
    description='A tool for generating track figures from video recordings of C. elegans',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
        'pandas',
        'joblib',
        'PySide6',
        'psutil',
    ],
    entry_points={
        'console_scripts': [
            'wormtrails-gui=wormtrails.gui:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='>=3.8',
)
