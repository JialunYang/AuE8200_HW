import numpy as np
import cv2

# Load a color image
img = cv2.imread('data/sets/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg')

# Show image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()