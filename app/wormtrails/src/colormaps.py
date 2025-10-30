import numpy as np
import math

white_to_black = []
for i in range(256):
    color = np.array([255-i,255-i,255-i]) # color shift from white to black
    white_to_black.append(color)
white_to_black = np.array(white_to_black)

black_to_white = []
for i in range(256):
    color = np.array([i,i,i]) # color shift from black to white
    black_to_white.append(color)
black_to_white = np.array(black_to_white)

blue_to_red = []
for i in range(256):
    color = np.array([255-i,0,i]) # color shift from blue to red
    blue_to_red.append(color)
blue_to_red = np.array(blue_to_red)

banded_blue_to_red = []
for i in range(256):
    brightness = math.sin(10*i*2*math.pi/255) # oscillate brightness 10x over colormap
    brightness = (brightness + 1)/2 # normalize to 0-1
    color = np.array([255-i,0,i]) # color shift from blue to red
    pixel_value = (color*brightness).astype(np.int64)
    banded_blue_to_red.append(pixel_value)
banded_blue_to_red = np.array(banded_blue_to_red)

dark_separated_blue_to_red = []
for i in range(256):
    brightness = math.ceil(math.sin(10*i*2*math.pi/255)) # toggle over time periods
    color = np.array([255-i,0,i]) # color shift from blue to red
    pixel_value = (color*brightness).astype(np.int64)
    dark_separated_blue_to_red.append(pixel_value)
dark_separated_blue_to_red = np.array(dark_separated_blue_to_red)