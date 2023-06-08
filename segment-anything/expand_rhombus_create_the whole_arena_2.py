import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

# Load the image
dir_pth = 'data/100005_data'
fns = glob.glob(os.path.join(dir_pth, '*.jpg'))
save_pth = os.path.join(dir_pth, 'results')
for fn in fns:
    if '1-9' in fn or '2-9' in fn or '2-12' in fn or '2-14' in fn:
        image = cv2.imread(fn)
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
        points = np.array(vertices, dtype=np.float32)

        cam_num = fn.split('/')[-1].split('_')[1]
        print(f'{cam_num}:  [{points[0]} {points[1]} {points[2]} {points[3]}')


        # Determine the width and height of the rectangle
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        width = int(x_max - x_min)
        height = int(y_max - y_min)

        # Define the desired width and height of the output rectangle
        output_width = 500
        output_height = 400

        # Define the 4 corner points of the output rectangle in clockwise order starting from top-left
        output_points = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(points, output_points)

        # Apply the perspective transformation to the image
        result = cv2.warpPerspective(image, matrix, (output_width, output_height))
        save_pth = os.path.join(save_pth, cam_num+'.jpg')
        cv2.imwrite(save_pth, result)
        # Display or save the resulting rectangle image
        cv2.imshow("Transformed Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
image size: 583, 1441
< regions>
1-1:   [[ 237.    7.] [1043.    4.] [1417.  580.] [  78.  580.]]   
1-1:  [[224.   7.] [1050.    5.] [1420.  576.] [ 49. 578.]]                       
1-2:   [[ 297.    7.] [ 997.    7.] [1356.  577.] [  52.  581.]]
1-2:  [[289.   8.] [993.   6.] [1356.  576.] [ 27. 576.]]
1-3:  [[288.   7.] [981.   7.] [1346.  577.] [  7. 578.]]   
1-4:  [[313.   8.] [987.   6.] [1360.  574.] [ 12. 578.]]
1-5:  [[294.   7.] [961.   6.] [1407.  578.] [ 53. 578.]]
1-6:  [[249.  11.] [959.   8.] [1374.  577.] [  4. 580.]]
1-7:  [[268.   6.] [988.   8.] [1403.  574.] [ 28. 578.]]
1-8:  [[183.   9.] [956.   5.] [1409.  578.] [  6. 579.]]
1-9:  [[260.   5.] [976.   4.] [1398.  578.] [ 38. 580.]]
1-10:  [[110.   6.] [949.   6.] [1409.  576.] [  4. 578.]]
1-11:  [[216.   8.] [944.   8.] [1368.  577.] [  6. 579.]]
1-12:  [[300.   7.] [971.   8.] [1355.  576.] [  9. 577.]]
1-13:  [[259.   7.] [1003.    7.] [1409.  577.] [ 33. 581.]]
1-14:  [[203.   9.] [983.   8.] [1388.  576.] [  6. 580.]]
1-15:  [[256.  10.] [956.   8.] [1377.  577.] [ 17. 578.]]
1-16:  [[234.   7.] [978.   5.] [1435.  579.] [ 36. 579.]]
2-1:  [[451.   8.] [1172.   11.] [1428.  577.] [  8. 579.]]
2-2:  [[522.   3.] [1167.    6.] [1440.  577.] [139. 578.]]
2-3:  [[496.   8.] [1205.   10.] [1438.  579.] [106. 578.]]
2-4:  [[446.   6.] [1173.    6.] [1437.  578.] [106. 581.]]
2-5:  [[419.   5.] [1145.    7.] [1437.  580.] [ 92. 580.]]
2-6:  [[396.   7.] [1130.   10.] [1429.  578.] [ 52. 578.]]
2-7:  [[428.   6.] [1123.    8.] [1437.  577.] [122. 577.]]
2-8:  [[391.   7.] [1061.   11.] [1371.  573.] [ 54. 579.]]
2-9:  [[506.   5.] [1149.    7.] [1438.  580.] [136. 579.]]
2-10:  [[479.   6.] [1134.    7.] [1419.  578.] [113. 579.]]
2-11:  [[392.   6.] [1095.    8.] [1432.  579.] [ 92. 578.]]
2-12:  [[449.   6.] [1108.    6.] [1425.  576.] [151. 577.]]
2-13:  [[358.   7.] [1075.    8.] [1393.  573.] [ 31. 578.]]
2-14:  [[395.   7.] [1058.    6.] [1378.  578.] [ 74. 579.]]
2-15:  [[426.   5.] [1159.    8.] [1418.  578.] [ 65. 578.]]
2-16:  [[347.   8.] [1109.    6.] [1351.  574.] [  6. 579.]]
'''