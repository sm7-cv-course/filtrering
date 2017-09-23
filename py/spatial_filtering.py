import cv2
import numpy as np
from matplotlib import pyplot as plt

"2D Convolution (Image Filtering) for image."

# Original image with 'Salt-and-pepper' noise addition
img = cv2.imread('./../images/Fig0318(b)(ckt-board-slt-pep-both-0pt2).tif', 0)

kernel = np.ones((5,5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Example of writting an image
out_path = './../out/average_img.png'
cv2.imwrite(out_path, dst)

blur = cv2.bilateralFilter(img, 9, 75, 75)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('./../images/Fig0316(a)(moon).tif', 0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_norm = np.zeros(laplacian.shape)
cv2.normalize(laplacian, laplacian_norm, 0, 255, cv2.NORM_MINMAX)

# Plot the results
plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(laplacian_norm, cmap = 'gray')
plt.title('laplacian_norm'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(img - laplacian_norm, cmap = 'gray')
plt.title('img - laplacian_norm'), plt.xticks([]), plt.yticks([])
plt.show()
