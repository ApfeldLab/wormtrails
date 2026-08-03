# Wormtrails 2.0
Wormtrails is a python package designed to create images and videos depicting the motion of C. elegans on solid media. Top-down darkfield or brightfield scans may be converted into stills or movies, with time encoded by color. This package was primarily designed to be used for chemotaxis, but it may be used to visualize locomotory patterns in a wide array of other behavioral assays, including lifespans.

## Installation
To install wormtrails, it is recommended to first install [anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install), then create and activate a python environment by executing the following commands in a shell prompt:
```
conda create -n wormtrails
conda activate wormtrails
```

After downloading this repository, execute the following commands from the same directory as this README.md file to install wormtrails.
```
conda install pip
pip install .
```

If they are not already installed, this will also install opencv, numpy, pandas, joblib, PySid6, and psutil, the requirements for wormtrails. 

## How to use wormtrails
With wormtrails 2.0, you can now use the GUI for most simple things. To use the GUI, run the command `wormtrails-gui` after activating yourconda environment with wormtrails installed. Wormtrails is still primarily intended for use through Python, but the GUI is a great way to get started with wormtrails.
In this repository, there is an examples directory with jupyter notebooks and a video of a chemotaxis control plate. To use it, install and run jupyter as so:
```
pip install jupyter
jupyter lab
```
This will let you access jupyter from a web browser, from which you can navigate to and open `make_trails.ipynb`

### Streaming videos from disk
If a video file is large, like is often the case with chemotaxis recordings, wormtrails can now stream it directly from disk while doing processing instead of loading the entire video into memory. In the GUI, this is done by selecting the `stream` option from the backend dropdown menu. Note that this is only compatible with average subtraction based motion, since linear model residual fitting requires the entire video to be loaded into memory. Some other functions have limited compatibility with the streaming option (see available functions in `src/wormtrails/streaming.py`).

### Preprocessing
The wormtrails processing pipeline consists of some preprocessing of a video scan, followed by the creation of the desired visualization. Most functions have an `in_place` parameter that can be set to `True` to overwrite the input array with the output of the preprocessing step, saving memory. 

Preprocessing involves correcting for vignetting in the raw capture (this may be skipped if vignetting is negligible), and subtracting an average frame from each frame of the scan (this may not be skipped). Vignetting correction uses either a median or Gaussian blur to approximate the background brightness. A custom frame range may be selected to be used to calculate the average, which is used as a reference for stationary background. If the parameters `average_start=0` and `average_end=-1` are used, then the entire scan will be used to create the background. This usually produces good results, unless some worms are stationary for a substantial proportion of the scan. In such a case, setting the `use_projection` parameter to `True` may help by using maximum/minimum projection depending on whether `light_background` is set to `True`/`False` instead of an average over time.

If vibrations have caused the plate to shift by a small amount (less than a pixel) over the course of the recording, and the recording is short enough to fit fully into memory, the `fit_pixel_linear_model` function may produce better results than `subtract_average`.

### Visualization parameters
The `show_time_encoding` function is used to preview your visualization, and takes the same inputs as `create_time_encoded_array` as well as using the same defaults. The only required input is an `average_subtracted_array` created during preprocessing, and by default it will create fully dark trails on a light background over a time window of 20 frames. Contrast may be increased by setting the `scale_factor` parameter to a value greater than 1. Noise floor in the scan may be corrected for by setting the `offset` parameter to a negative value. The `window` parameter can be set to a value of 1 to not show any trails, or any value up to the total number of frames in the scan. For the `create_time_encoded_frame` function, a specific `start_time` must be given as a frame index. Note that if `fit_pixel_linear_model` was used in preprocessing, the appropriate value for `scale_factor` is often lower.

The final and most versatile visualization parameter is `colormap`. This parameter must be in the form of a two dimensional numpy array, with each column containing a desired BGR (Blue Green Red) color value. These colors will be applied to the trails in order from their start frame to end frame, with the first column being used to color the first frame and the last column being used to color the last frame, and colors in between being chosen from the middle columns. 

Several colormaps are included, such as white-to-black and black-to-white gradients, as well as some oscillating brightness colormaps. They can be found in `src/wormtrails/colormaps.py` and accessed the same as wormtrails functions. Colormaps in svg form may be used by first converting them to a pixel-based image format such as .png with image editing software such as GIMP, then loading them in as so:
```
import wormtrails as wts
import cv2
import numpy as np

colormap = cv2.imread("./hue_ramps_16.png").astype(np.int64)[0,:,:]
```

Single colors may be applied as a colormap by their BGR value. For example, to color trails green, the parameter `colormap=np.array([[0,255,0]])` may be used for any visualization function.
