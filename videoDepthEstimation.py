import cv2
import numpy as np
import playsound as ps
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator

depthEstimator = midasDepthEstimator()
cap = cv2.VideoCapture(0)
cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL) 	

while cap.isOpened():
	ret, img = cap.read()

	if ret:
		# Estimate depth
		colorDepth = depthEstimator.estimateDepth(img)

		colorDepth = cv2.cvtColor(colorDepth, cv2.COLOR_BGR2GRAY)

		height, width = colorDepth.shape[:2]

		w, h = (8, 8)

		temp = cv2.resize(colorDepth, (w, h), interpolation=cv2.INTER_LINEAR)

		output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
		
		hsv_frame = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
		low = np.array([220, 220, 220])
		high = np.array([255, 255, 255])
		object_mask = cv2.inRange(hsv_frame,low, high)
		object_edge = cv2.bitwise_and(output,output,mask=object_mask)

		edges= cv2.Canny(object_edge, 50,200)

		contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		number_of_objects_in_image= len(contours)

		if(number_of_objects_in_image):
			ps.playsound('Blindwave.wav')

		# cv2.imshow("Depth Image", output)
		cv2.imshow("Depth Image",object_edge)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
