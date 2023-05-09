import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("test.jpg", 0)

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.xticks([])
plt.yticks([])

# Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
sobel = cv2.convertScaleAbs(sobel)
plt.subplot(2, 3, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.xticks([])
plt.yticks([])

# Robert
robertx = np.array([[-1, 0], [0, 1]], dtype=float)
roberty = np.array([[0, -1], [1, 0]], dtype=float)
robertx_img = cv2.filter2D(img, -1, robertx)
roberty_img = cv2.filter2D(img, -1, roberty)
robert = cv2.addWeighted(robertx_img, 0.5, roberty_img, 0.5, 0)
plt.subplot(2, 3, 3)
plt.imshow(robert, cmap="gray")
plt.title("Robert")
plt.xticks([])
plt.yticks([])

# Prewitt
prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
prewittx_img = cv2.filter2D(img, -1, prewittx)
prewitty_img = cv2.filter2D(img, -1, prewitty)
prewitt = cv2.addWeighted(prewittx_img, 0.5, prewitty_img, 0.5, 0)
plt.subplot(2, 3, 4)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")
plt.xticks([])
plt.yticks([])

# Canny
canny = cv2.Canny(img, 100, 200)
plt.subplot(2, 3, 5)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.xticks([])
plt.yticks([])

# LOG
log = cv2.Laplacian(img, cv2.CV_64F)
log = cv2.convertScaleAbs(log)
plt.subplot(2, 3, 6)
plt.imshow(log, cmap="gray")
plt.title("LOG")
plt.xticks([])
plt.yticks([])

plt.show()
