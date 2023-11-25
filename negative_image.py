import numpy as np
import cv2

# Read the image in grayscale
img = cv2.imread('Images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Display the input image
cv2.imshow('Input Image', img)

# Loop through each pixel and calculate the negative
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img.itemset((i, j), 255 - img.item(i, j))

# Display the negative image
cv2.imshow('Negative Image', img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
