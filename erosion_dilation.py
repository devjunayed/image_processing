import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('Images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image.")
else:
    # Display the input image
    cv2.imshow('Input Image', img)

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Function for custom erosion
    def custom_erosion(image, kernel):
        rows, cols = image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                # Erosion: minimum value in the neighborhood
                result[i, j] = np.min(image[i-2:i+3, j-2:j+3] * kernel)

        return result

    # Function for custom dilation
    def custom_dilation(image, kernel):
        rows, cols = image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                # Dilation: maximum value in the neighborhood
                result[i, j] = np.max(image[i-2:i+3, j-2:j+3] * kernel)

        return result

    # Apply custom erosion and display the result
    erosion_img = custom_erosion(img, kernel)
    cv2.imshow('Custom Erosion', erosion_img)

    # Apply custom dilation and display the result
    dilation_img = custom_dilation(img, kernel)
    cv2.imshow('Custom Dilation', dilation_img)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
