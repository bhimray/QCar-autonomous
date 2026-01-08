## task_lane_following.py

from pal.utilities.vision import Camera2D
from pal.products.qcar import QCar
from pal.utilities.math import Filter
from pal.utilities.probe import Probe
from hal.utilities.image_processing import ImageProcessing

import time
import numpy as np
import cv2
import math

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing Parameters and methods
sampleRate = 60
sampleTime = 1/sampleRate
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Additional parameters
imageWidth  = 1640
imageHeight = 820
cameraID 	= "3@tcpip://localhost:18964"

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#Setting Filter
steeringFilter = Filter().low_pass_first_order_variable(25, 0.033)
next(steeringFilter)
dt = 0.033

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Initialize the CSI cameras
myCam = Camera2D(cameraId=cameraID, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## QCar and probe Initialization
myCar = QCar(readMode=1, frequency=60)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Main Loop
try:
	while True:
		start = time.time()
		# Capture RGB Image from CSI
		myCam.read()
		# Crop out a piece of the RGB to improve performance
		croppedRGB = myCam.imageData[524:674, 0:820]

		# Convert to HSV and then threshold it for yellow
		hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)
		binaryImage = ImageProcessing.binary_thresholding(frame= hsvBuf,
													lowerBounds=np.array([10, 50, 100]),
													upperBounds=np.array([45, 255, 255]))
        
		# Overlay detected yellow lane over raw RGB image
		binaryImage=binaryImage/255
		processed = myCam.imageData
		processed[524:674, 0:820,2]=processed[524:674, 0:820,2]+(255-processed[524:674, 0:820,2])*binaryImage
		processed[524:674, 0:820,1]=processed[524:674, 0:820,1]*(1-binaryImage)
		processed[524:674, 0:820,0]=processed[524:674, 0:820,0]*(1-binaryImage)

		# Display the RGB cropped RGB Image and the resized binary image
		cv2.imshow('Detection Overlay', cv2.resize(processed, (820,410)) )
		cv2.imshow('Binary Image', cv2.resize(hsvBuf, (820,410)) )

		# Find slope and intercept of linear fit from the binary image
		slope, intercept = ImageProcessing.find_slope_intercept_from_binary(binary=binaryImage)

		# steering from slope and intercept
		rawSteering = 1.5*(slope - 0.3419) + (1/150)*(intercept+5)
		steering = steeringFilter.send((np.clip(rawSteering, -0.5, 0.5), dt))

		LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
		myCar.read_write_std(0.01, steering, LEDs)
		cv2.waitKey(1)
		end = time.time()
		dt = end - start

except KeyboardInterrupt:
	print("User interrupted!")

finally:
	# Terminate camera and QCar
	myCam.terminate()
	myCar.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
