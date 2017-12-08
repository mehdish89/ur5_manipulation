# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
import imutils
import cv2
import numpy as np

from matplotlib import pyplot as plt

import pyqtgraph as pg

import time

from scipy.signal import find_peaks_cwt

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
# image = cv2.imread(args["image"])

def find_peaks(a):
  x = np.array(a)
  max = np.max(x)
  lenght = len(a)
  ret = []
  for i in range(lenght):
      ispeak = True
      if i-1 > 0:
          ispeak &= (x[i] > 1.8 * x[i-1])
      if i+1 < lenght:
          ispeak &= (x[i] > 1.8 * x[i+1])

      ispeak &= (x[i] > 0.05 * max)
      if ispeak:
          ret.append(i)
  return ret

# plotWidget = pg.plot(title="Three plot curves")

def closest(set, val):
	set = np.array(set)
	index = np.argmin(abs(set-val))
	return set[index]


def detect_boxes(image, draw = False):

	# resized = imutils.resize(image, width=600)
	# ratio = image.shape[0] / float(resized.shape[0])

	rects = []

	ratio = 1.
	resized = image

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it

	resized = cv2.GaussianBlur(resized, (11, 11), 0)

	# hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)


	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
	# gray = np.array(hsv[:,:,2])


	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]

	# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 #            cv2.THRESH_BINARY,201, -10)


	mask = thresh.copy()

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)


	kernel = np.ones((7,7),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	

	

	thresh = mask.copy()

	mask = thresh / 255

	rgb = resized

	# h = np.array(hsv[:,:,0])
	# s = np.array(hsv[:,:,1])
	# v = np.array(hsv[:,:,2])

	# hist = cv2.calcHist([blurred],[0],None,[256],[0,256])
	# hist_h = cv2.calcHist([h],[0],None,[256],[0,256]).reshape(-1)
	# hist_s = cv2.calcHist([s],[0],None,[256],[0,256]).reshape(-1)
	# hist_v = cv2.calcHist([v],[0],None,[256],[0,256]).reshape(-1)
	
	# calcBackProject
	# plotWidget.clear()
	# plotWidget.plot(hist_h, pen='b')
	# plotWidget.plot(hist_s, pen='g')
	# plotWidget.plot(hist_v, pen='r')
	# plotWidget.show()
	# return

	# hist_r = hist_r.reshape(-1)


	# h_th = 11
	# s_th = 243
	# v_th = 210

	# h_peaks = find_peaks_cwt(hist_h, np.arange(1,10))
	# s_peaks = find_peaks_cwt(hist_s, np.arange(1,10))
	# v_peaks = find_peaks_cwt(hist_v, np.arange(1,50))

	# h_th = closest(h_peaks, h_th)
	# s_th = closest(s_peaks, s_th)
	# v_th = closest(v_peaks, v_th)

	# print(h_th, s_th, v_th)

	# hsv_mid = np.array([h_th, s_th, v_th]) 
	# hsv_range = np.array([5, 15, 45])

	# hsv_min = hsv_mid - hsv_range
	# hsv_max = hsv_mid + hsv_range

	# [  6.  97.  67.]
	# [  24.  255.  253.]

	hsv_min = np.array([0, 120, 67])
	hsv_max = np.array([20, 255, 253])

	mask = cv2.inRange(hsv, hsv_min, hsv_max)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)


	kernel = np.ones((7,7),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	# print(h_peaks)
	# print(s_peaks)
	# print(v_peaks)
	# print(find_peaks(hist))

	

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)

	# return

	

	# cv2.waitKey(0)

	thresh = mask.copy()

	masked = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)	
	mix = rgb/2 + masked/2


	# cv2.imshow("Image", masked)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)

	# return
	


	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# sd = ShapeDetector()


	# loop over the contours

	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)

		try:
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
		except:
			continue

		cnt = c
		rect = cv2.minAreaRect(cnt)
		# print(rect)
		# box = cv2.boxPoints(rect)
		box = cv2.cv.BoxPoints(rect)

		tbox = np.array(box)
		tbox[:,[0, 1]] = tbox[:,[1, 0]]
		rects.append(tbox)

		box = np.int0(box)
		cv2.drawContours(resized,[box],0,(0,0,255),2)

		# print(len(cnt))

		# cv2.drawContours(resized,[cnt],0,(0,0,255),2)



		# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		# cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		# 	0.5, (255, 255, 255), 2)

	# draw = True

	if draw:

		image = np.fmax(resized, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))



		cv2.imshow("Image", image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			exit(0)

	return rects


if __name__ == "__main__":
	cap = cv2.VideoCapture(2)

	while True:

		ret, frame = cap.read()
		image = frame
		detect_boxes(image, draw = True)
		# time.sleep(0.2)
