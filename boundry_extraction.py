import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('Images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Display the input image
cv2.imshow('Input Image', img)

# Function for custom Sobel operator
def custom_sobel(image):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    # Sobel operators for horizontal and vertical edges
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Convolve the image with Sobel operators
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
    
    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize the result to the range [0, 255]
    result = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    return result

# Apply custom Sobel operator for boundary extraction
boundary_img = custom_sobel(img)
cv2.imshow('Boundary Extraction', boundary_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
