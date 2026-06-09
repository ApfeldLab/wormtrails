"""
Predefined colormaps for time-encoded trail visualization.

Each colormap is a 2D NumPy array of shape (N, 3) where N is the maximum window size.
Colors are BGR values in the range [0, 255].

white_to_black: Smooth gradient from white (255,255,255) to black (0,0,0)
black_to_white: Smooth gradient from black to white
blue_to_red: Smooth gradient from blue to red
banded_blue_to_red: Blue to red gradient with oscillating brightness
dark_separated_blue_to_red: Discrete blue to red gradient with alternating frames
"""

import numpy as np
import cv2

__all__ = [
    'black',
    'white',
    'white_to_black',
    'black_to_white',
    'blue_to_red',
    'banded_blue_to_red',
    'dark_separated_blue_to_red',
    'middle_grey_last_black',
    'hsv_rainbow',
]

black = np.array([[0,0,0]], dtype=np.float64)

white = np.array([[255,255,255]], dtype=np.float64)

white_to_black = []
for i in range(256):
    color = np.array([255-i,255-i,255-i], dtype=np.float64)
    white_to_black.append(color)
white_to_black = np.array(white_to_black, dtype=np.float64)


black_to_white = []
for i in range(256):
    color = np.array([i,i,i], dtype=np.float64)
    black_to_white.append(color)
black_to_white = np.array(black_to_white, dtype=np.float64)


blue_to_red = []
for i in range(256):
    color = np.array([255-i,0,i], dtype=np.float64)
    blue_to_red.append(color)
blue_to_red = np.array(blue_to_red, dtype=np.float64)


banded_blue_to_red = []
for i in range(256):
    brightness = np.sin(10*i*2*np.pi/255)
    brightness = (brightness + 1)/2
    color = np.array([255-i,0,i], dtype=np.float64)
    pixel_value = (color*brightness)
    banded_blue_to_red.append(pixel_value)
banded_blue_to_red = np.array(banded_blue_to_red, dtype=np.float64)


dark_separated_blue_to_red = []
for i in range(256):
    brightness = np.ceil(np.sin(10*i*2*np.pi/255))
    color = np.array([255-i,0,i], dtype=np.float64)
    pixel_value = (color*brightness)
    dark_separated_blue_to_red.append(pixel_value)
dark_separated_blue_to_red = np.array(dark_separated_blue_to_red, dtype=np.float64)


middle_grey_last_black = []
window = 20
for i in range(window-1):
    middle_grey_last_black.append(np.array([128, 128, 128], dtype=np.float64))
middle_grey_last_black.append(np.array([0, 0, 0], dtype=np.float64))
middle_grey_last_black = np.array(middle_grey_last_black, dtype=np.float64)


hsv_rainbow = []
for i in range(180):
    hsv = np.uint8([[[i, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    hsv_rainbow.append(np.array([bgr[0], bgr[1], bgr[2]], dtype=np.float64))
hsv_rainbow = np.array(hsv_rainbow, dtype=np.float64)
