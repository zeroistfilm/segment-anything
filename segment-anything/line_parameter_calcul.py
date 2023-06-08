import cv2
import numpy as np

# Global variables to store the coordinates of the two points
point1 = (-1, -1)
point2 = (-1, -1)

def on_mouse(event, x, y, flags, param):
    global point1, point2

    if event == cv2.EVENT_LBUTTONDOWN:
        if point1 == (-1, -1):
            point1 = (x, y)
            print(f"First point coordinates: x={x}, y={y}")
        else:
            point2 = (x, y)
            print(f"Second point coordinates: x={x}, y={y}")
            calculate_line_parameters()

def calculate_line_parameters():
    global point1, point2

    # Calculate the line parameters
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    slope = dy / dx if dx != 0 else np.inf
    intercept = point1[1] - slope * point1[0]

    print(f"Line parameters: slope={slope}, intercept={intercept}")

# Load the image
image = cv2.imread('data/172925_1-10_7387700_2.jpg')

# Create a window and display the image
cv2.namedWindow("Image")
cv2.imshow("Image", image)

# Set the mouse callback function
cv2.setMouseCallback("Image", on_mouse)

# Wait for a key press
cv2.waitKey(0)

# Destroy the window
cv2.destroyAllWindows()

'''
-- wall side
First point coordinates: x=114, y=7
Second point coordinates: x=10, y=552
Line parameters: slope=-5.240384615384615, intercept=604.4038461538461
-- egg side
First point coordinates: x=904, y=5
Second point coordinates: x=1329, y=577
Line parameters: slope=1.3458823529411765, intercept=-1211.6776470588236

'''