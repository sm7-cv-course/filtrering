import cv2
import numpy as np
from matplotlib import pyplot as plt

"2D Convolution (Image Filtering) for image."

# Original image with 'Salt-and-pepper' noise addition
img = cv2.imread('./../images/shirt_video.jpg',0)

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
