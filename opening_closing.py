import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('Images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Display the input image
cv2.imshow('Input Image', img)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Function for custom erosion
def custom_erosion(image, kernel):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Erosion: minimum value in the neighborhood
            result[i, j] = np.min(image[i-1:i+2, j-1:j+2] * kernel[1:-1, 1:-1])

    return result

# Function for custom dilation
def custom_dilation(image, kernel):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Dilation: maximum value in the neighborhood
            result[i, j] = np.max(image[i-1:i+2, j-1:j+2] * kernel[1:-1, 1:-1])

    return result

# Function for custom opening
def custom_opening(image, kernel):
    # Opening: erosion followed by dilation
    opened_img = custom_dilation(custom_erosion(image, kernel), kernel)
    return opened_img

# Function for custom closing
def custom_closing(image, kernel):
    # Closing: dilation followed by erosion
    closed_img = custom_erosion(custom_dilation(image, kernel), kernel)
    return closed_img

# Apply custom opening and display the result
opened_img = custom_opening(img, kernel)
cv2.imshow('Custom Opening', opened_img)

# Apply custom closing and display the result
closed_img = custom_closing(img, kernel)
cv2.imshow('Custom Closing', closed_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
