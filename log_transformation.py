import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('Images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Display the input image
cv2.imshow('Input Image', img)

# Function for custom log transformation
def custom_log_transform(image):
    # Add a small constant to avoid log(0)
    constant = 1
    
    # Apply log transformation to each pixel
    result = np.log1p(image + constant)
    
    # Normalize the result to the range [0, 255]
    result = (result / result.max() * 255).astype(np.uint8)
    
    return result

# Apply custom log transformation
log_transformed_img = custom_log_transform(img)
cv2.imshow('Log Transformation', log_transformed_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
