"""FLIR framesplit timelapse capture used in Apfeld lab's implementation.

Restrictions / assumptions:
- Requires exactly one Spinnaker-compatible FLIR camera; otherwise the
  script exits without releasing the PySpin system instance.
- Requires PySpin and OpenCV. OpenCV must support the FFV1 codec, which
  stock macOS builds often lack. VideoWriter success is not checked, so
  an unsupported codec can silently produce an empty/unreadable file.
- Assumes Mono8 (grayscale) frames. PixelFormat is never set, so a camera
  defaulting to Bayer/RGB will break the GRAY2BGR conversion and averaging.
- Requires the following to be available/writeable: ExposureAuto, 
  ExposureMode, ExposureTime, GainAuto, Gain, AcquisitionMode, 
  StreamBufferHandlingMode
- Output fps is hardcoded to 60.0 in VideoWriter.
- Only tested on a 4000x3000 sensor.
- Region size is input_width // cols by input_height // rows; leftover
  edge pixels are silently dropped when the sensor size is not divisible.
- --regions accepts only "A-B" or "a,b,c" (no combined "0-5,7" form) and
  indices are not bounds-checked; out-of-range or negative indices produce
  empty/wrapped crops. --remap must match the capture count and contain
  unique values; duplicates make two writers target the same file.
- -i/--interval must be > 0 and -d/--duration, -e/--exposure must be
  positive.
- Preview blocks until SPACE is pressed. Closing the window with
  the X button does not exit preview.
- Script-level argparse + os.chdir run on import (no __main__ guard), so
  this file is not importable. The positional dir must already exist and
  parent dirs for -o must already exist. Existing per-region outputs raise
  FileExistsError; the extension is passed through to VideoWriter unchecked.
"""

import PySpin
import cv2
import numpy as np
import time
import sys
import os
import threading
import queue
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dir', nargs='?', default=None, help='Directory to chdir into')
parser.add_argument('-o', '--out', default='raw.avi', help='Output video file name base')
parser.add_argument('-e', '--exposure', type=int, default=23000, help='Exposure time in microseconds (default: 23000)')
parser.add_argument('-d', '--duration', type=float, default=60.0, help='Duration of capture in seconds (default: 60)')
parser.add_argument('-i', '--interval', type=float, default=1.0, help='Capture interval in seconds per frame (default: 1)')
parser.add_argument('-c', '--cols', type=int, default=4, help='Number of columns (default 4)')
parser.add_argument('-r', '--rows', type=int, default=3, help='Number of rows (default 3)')
parser.add_argument('--regions', type=str, default='', help='Indices to capture (e.g., "0-5" or "0,2,4"). Default is all.')
parser.add_argument('--remap', type=str, default='', help='Remap indices (e.g., "5,6,7"). Overrides --start-index.')
parser.add_argument('--start-index', type=int, default=None, help='Start index for numbering output files.')
args_parsed = parser.parse_args()

if args_parsed.dir:
	os.chdir(args_parsed.dir)

output_filename = args_parsed.out
exposure_time = args_parsed.exposure
duration = args_parsed.duration
interval = args_parsed.interval
cols = args_parsed.cols
rows = args_parsed.rows

num_regions = cols * rows
if args_parsed.regions:
	if '-' in args_parsed.regions:
		start_reg, end_reg = map(int, args_parsed.regions.split('-'))
		capture_indices = list(range(start_reg, end_reg + 1))
	else:
		capture_indices = [int(x) for x in args_parsed.regions.split(',')]
else:
	capture_indices = list(range(num_regions))

if args_parsed.remap:
	remap_indices = [int(x) for x in args_parsed.remap.split(',')]
	if len(remap_indices) != len(capture_indices):
		print("Error: length of remap must match number of captured regions.")
		sys.exit(1)
elif args_parsed.start_index is not None:
	remap_indices = list(range(args_parsed.start_index, args_parsed.start_index + len(capture_indices)))
else:
	remap_indices = capture_indices

index_map = dict(zip(capture_indices, remap_indices))

system = PySpin.System.GetInstance()
version = system.GetLibraryVersion()

cam_list = system.GetCameras()
num_cameras = cam_list.GetSize()

if num_cameras == 1:
	cam = cam_list.GetByIndex(0)
else:
	print("either no cameras or multiple cameras detected")
	quit()

nodemap_tldevice = cam.GetTLDeviceNodeMap()
cam.Init()
nodemap = cam.GetNodeMap()

# setup buffer
global continue_recording
continue_recording = True
sNodemap = cam.GetTLStreamNodeMap()
node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
node_newestonly_mode = node_newestonly.GetValue()
node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

# set camera parameters
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
cam.ExposureTime.SetValue(exposure_time) # exposure time in microseconds
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
cam.Gain.SetValue(0) # gain 0 dB

# setup acquisition
node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
cam.BeginAcquisition() # start
device_serial_number = ''
node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

# get and display preview
cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('camera', 800, 800)
while(continue_recording):
	image_result = cam.GetNextImage()
	image_data = image_result.GetNDArray()
	input_height, input_width = image_data.shape[:2]
	region_w = input_width // cols
	region_h = input_height // rows

	display = image_data.copy()
	for r in range(1, rows):
		y = r * region_h
		display[max(0, y-5):min(input_height, y+5), :] = 0
	for c in range(1, cols):
		x = c * region_w
		display[:, max(0, x-5):min(input_width, x+5)] = 0

	cv2.imshow('camera', display)
	if cv2.waitKey(1) == ord(' '):
		cv2.destroyAllWindows()
		continue_recording = False
	image_result.Release()

interval_stack = []
all_timestamps = []
interval_timestamps = []

# setup writers
base, ext = os.path.splitext(output_filename)
writers = {}
fourcc = cv2.VideoWriter_fourcc(*'FFV1')

for idx in capture_indices:
	out_name = f"{base}_{index_map[idx]}{ext}"
	if Path(out_name).is_file():
		raise FileExistsError(f"The file {out_name} already exists")
	writers[idx] = cv2.VideoWriter(out_name, fourcc, 60.0, (region_w, region_h))

frame_queue = queue.Queue()

def process_and_save_frames():
	while True:
		item = frame_queue.get()
		if item is None:
			break
		mean_frame = np.mean(item, axis=0).astype(np.uint8)
		mean_frame_color = cv2.cvtColor(mean_frame, cv2.COLOR_GRAY2BGR)

		for idx in capture_indices:
			r = idx // cols
			c = idx % cols
			y1 = r * region_h
			y2 = y1 + region_h
			x1 = c * region_w
			x2 = x1 + region_w
			crop = mean_frame_color[y1:y2, x1:x2]
			writers[idx].write(crop)

		frame_queue.task_done()

writer_thread = threading.Thread(target=process_and_save_frames)
writer_thread.start()

start_time = time.time()
while(time.time() < start_time + duration):
	cap_time = time.time()
	cap_sec = cap_time - start_time
	image_result = cam.GetNextImage()
	image_data = image_result.GetNDArray()
	# Create a copy so the memory isn't overwritten by the camera while processing in another thread
	frame = image_data.copy()
	image_result.Release()

	if len(interval_stack) == 0:
		interval_stack.append(frame)
	elif np.floor(cap_sec/interval)*interval == np.floor(all_timestamps[-1]/interval)*interval:
		interval_stack.append(frame)
	else:
		frame_queue.put(interval_stack)
		interval_timestamps.append(cap_sec)
		interval_stack = [frame]

	print(cap_sec, end='\r')
	all_timestamps.append(cap_sec)

if len(interval_stack) > 0:
	frame_queue.put(interval_stack)

frame_queue.put(None)
writer_thread.join()

print("framerate: ", len(all_timestamps)/duration, "fps")

for writer in writers.values():
	writer.release()

cv2.destroyAllWindows()

# Delete NumPy views of PySpin memory before tearing down the system
if 'image_data' in locals():
	del image_data
if 'image_result' in locals():
	del image_result

cam.EndAcquisition()
cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
