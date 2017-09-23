import cv2
import numpy as np
from matplotlib import pyplot as plt

"Median filtering for 2d image."

# Original image with 'Salt-and-pepper' noise addition
img = cv2.imread('./../images/Fig0318(b)(ckt-board-slt-pep-both-0pt2).tif',0)

# Median filtering
img_median3 = cv2.medianBlur(img, 3)
img_median5 = cv2.medianBlur(img, 5)
img_median7 = cv2.medianBlur(img, 7)

# Plot the results
plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(img_median3, cmap = 'gray')
plt.title('Median3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(img_median5, cmap = 'gray')
plt.title('Median5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(img_median7, cmap = 'gray')
plt.title('Median7x7'), plt.xticks([]), plt.yticks([])
plt.show()
