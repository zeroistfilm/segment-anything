import cv2
import numpy as np
import glob, os
# Global variables
points = []

def on_mouse(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Image", image)

# Load the image
# fn = 'data/bins_only/161525_2-11_feeding_bins_only_woLegend.png'
# fn = 'data/155006_2-11_feeding.jpg'
# fn = 'data/110008_2-11_etc.jpg'
dir_pth = 'data/100005_data'
fns = glob.glob(os.path.join(dir_pth, '*.jpg'))
# image = cv2.imread('data/155006_2-11_feeding.jpg')
roi_foodbin_slate_0_1 = {}
roi_foodbin_ground_2 = {} # ground - slate side bins
roi_foodbin_ground_3 = {} # ground - wall side bins
roi_foodbin_wall = {}  # packed to wall
roi_drink = {}

for i, fn in enumerate(fns):
    image = cv2.imread(fn)
    cam_num = fn.split('_')[2]
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
    roi_foodbin_slate_0_1[cam_num] = points


# Create a mask with the selected polygon
mask = np.zeros_like(image[:, :, 0])
print(points)
cv2.fillPoly(mask, [np.array(points)], 255)

# Check if a point is inside the selected polygon
point = (1, 1)  # Example point, replace with your own coordinates
is_inside = cv2.pointPolygonTest(np.array(points), point, False)

# is_inside > 0: Point is inside the polygon
# is_inside == 0: Point is on the polygon boundary
# is_inside < 0: Point is outside the polygon
if is_inside > 0:
    print("Point is inside the polygon")
elif is_inside == 0:
    print("Point is on the polygon boundary")
else:
    print("Point is outside the polygon")

'''
polygon points (points of interest)
data/172925_1-10_7387700_2.jpg:  [(112, 7), (903, 7), (1325, 580), (6, 580)]
data/155006_2-11_feeding.jpg:  [(422, 5), (1093, 12), (1423, 581), (164, 581)]
    - feed_1_2_line region: [(503, 6), (696, 4), (680, 577), (335, 579)]
    - feed_3_line region:  [(730, 7), (873, 6), (986, 579), (725, 578)]
    - feed_4_line region:  [(878, 3), (1003, 7), (1129, 272), (967, 274)]
    - drinking: [(421, 5), (522, 5), (356, 580), (168, 581)]
    
'''