import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
# Load the image
fn = 'data/100005_data/100005_1-2_5607888_1.jpg'
image = cv2.imread(fn)

# Define the vertices of the rhombus (assumed order: top, right, bottom, left)
# Global variables
vertices = []

def on_mouse(event, x, y, flags, param):
    global vertices

    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Image", image)
clone = image.copy()
# Create a window and display the image
cv2.namedWindow("Image")
cv2.imshow("Image", image)
# Set the mouse callback function
cv2.setMouseCallback("Image", on_mouse)
# Wait for the user to select points
cv2.waitKey(0)
# Close the window
cv2.destroyAllWindows()
vertices = np.array(vertices, dtype=np.float32)
print(vertices)
# Determine the width and height of the rectangle
x_min = np.min(vertices[:, 0])
x_max = np.max(vertices[:, 0])
y_min = np.min(vertices[:, 1])
y_max = np.max(vertices[:, 1])
width = int(x_max - x_min)
height = int(y_max - y_min)

# Create a new blank image with the dimensions of the rectangle
new_image = np.zeros((height, width, 3), dtype=np.uint8)
# ========================== warp method ====================================================
# Create a transformation matrix for perspective transformation
src_points = np.float32(vertices)
dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply perspective transformation to copy the rhombus region to the rectangle
cv2.warpPerspective(image, M, (width, height), new_image)
# ============================== affine method ================================================
# # Create an affine transformation matrix
# src_points = np.float32(vertices)
# dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
# M = cv2.getAffineTransform(src_points[:3], dst_points)
#
# # Apply affine transformation to copy the rhombus region to the rectangle
# cv2.warpAffine(image, M, (width, height), new_image)
# ============================================================================================
# Display or save the resulting rectangle image
cv2.imshow("Expanded Rectangle", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ============================================================================================
