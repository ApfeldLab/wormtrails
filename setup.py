from setuptools import setup, find_packages

setup(
    name='wormtrails',
    version='0.1.0',
    author='Christopher Dante Ashih',
    description='A tool for generating track figures from video recordings of C. elegans',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/microscope_figures',  # Update with your GitHub URL
    packages=find_packages(where='app'),
    package_dir={'': 'app'},
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='>=3.8',
)
