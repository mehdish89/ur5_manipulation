# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
# image = cv2.imread(args["image"])


def detect_rects(image, draw = False):

	# resized = imutils.resize(image, width=600)
	# ratio = image.shape[0] / float(resized.shape[0])

	rects = []

	ratio = 1.
	resized = image

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


	mask = thresh.copy()

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)

	# cv2.imshow("Image", mask)

	thresh = mask.copy()

	# cv2.waitKey(0)

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)

	# continue


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
		box = cv2.boxPoints(rect)

		rects.append(box)

		box = np.int0(box)
		cv2.drawContours(image,[box],0,(0,0,255),2)



		# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		# cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		# 	0.5, (255, 255, 255), 2)


	if draw:
		cv2.imshow("Image", image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			exit(0)

	return rects


cap = cv2.VideoCapture(2)

while True:

	ret, frame = cap.read()
	image = frame
	print(detect_rects(image))