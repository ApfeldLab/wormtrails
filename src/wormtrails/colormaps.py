import numpy as np
import math


"""
Predefined colormaps for time-encoded trail visualization.

Each colormap is a 2D NumPy array of shape (N, 3) where N is the maximum window size.
Colors are RGB values in the range [0, 255].

white_to_black: Smooth gradient from white (255,255,255) to black (0,0,0)
black_to_white: Smooth gradient from black to white
blue_to_red: Smooth gradient from blue to red
banded_blue_to_red: Blue to red gradient with oscillating brightness
dark_separated_blue_to_red: Discrete blue to red gradient with alternating frames
"""

white_to_black = []
for i in range(256):
    color = np.array([255-i,255-i,255-i])
    white_to_black.append(color)
white_to_black = np.array(white_to_black)


black_to_white = []
for i in range(256):
    color = np.array([i,i,i])
    black_to_white.append(color)
black_to_white = np.array(black_to_white)


blue_to_red = []
for i in range(256):
    color = np.array([255-i,0,i])
    blue_to_red.append(color)
blue_to_red = np.array(blue_to_red)


banded_blue_to_red = []
for i in range(256):
    brightness = math.sin(10*i*2*math.pi/255)
    brightness = (brightness + 1)/2
    color = np.array([255-i,0,i])
    pixel_value = (color*brightness).astype(np.int64)
    banded_blue_to_red.append(pixel_value)
banded_blue_to_red = np.array(banded_blue_to_red)


dark_separated_blue_to_red = []
for i in range(256):
    brightness = math.ceil(math.sin(10*i*2*math.pi/255))
    color = np.array([255-i,0,i])
    pixel_value = (color*brightness).astype(np.int64)
    dark_separated_blue_to_red.append(pixel_value)
dark_separated_blue_to_red = np.array(dark_separated_blue_to_red)