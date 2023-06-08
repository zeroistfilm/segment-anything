import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

def main():
    if preproc == 0:
        # Load the image
        dir_pth = 'data/100005_data'
        fns = glob.glob(os.path.join(dir_pth, '*.jpg'))

        for fn in fns:
            image = cv2.imread(fn)
            clone = image.copy()
            cam_num = fn.split('/')[-1].split('_')[1]
            points = np.asarray(regions[cam_num], dtype=np.float32)
            save_pth = os.path.join(dir_pth, 'results')
            # print(f'{cam_num}:  [{points[0]} {points[1]} {points[2]} {points[3]}')


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
            if len(cam_num) == 3: cam_num = cam_num[:-1] + '0' + cam_num[-1]
            save_pth = os.path.join(save_pth, cam_num+'.jpg')
            cv2.imwrite(save_pth, result)
            # Display or save the resulting rectangle image
            # cv2.imshow("Transformed Image", result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    if preproc == 1:

        # Load the 10 images (replace "image1.jpg" to "image10.jpg" with the actual image filenames)
        images = []
        # Load the image
        dir_pth = 'data/100005_data/results'
        fns = glob.glob(os.path.join(dir_pth, '*.jpg'))
        fns_sorted = sorted(fns)
        images = {}
        for i, fn in enumerate(fns_sorted):
            print(fn)
            img = cv2.imread(fn)
            # images.append(img)
            images[i] = (img, fn)
        # Create a blank canvas to hold the combined image
        combined_height = images[0][0].shape[0] * 16  # Height of the combined image
        combined_width = images[0][0].shape[1] * 2 # Width of the combined image
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Iterate over the images and place them in the combined image
        for i in range(len(list(images.keys()))):
            # print(i)
            # row = i // 2  # Calculate the row index
            # col = i % 2  # Calculate the column index
            if i < 16:
                row = int(images[i][1].split('/')[-1].split('.')[0].split('-')[1]) - 1  # Calculate the row index
                col = int(images[i][1].split('/')[-1].split('.')[0].split('-')[0]) - 1  # Calculate the column index
            else:
                row = 16 - int(images[i][1].split('/')[-1].split('.')[0].split('-')[1])  # Calculate the row index
                col = int(images[i][1].split('/')[-1].split('.')[0].split('-')[0]) - 1  # Calculate the column index
            x = col * images[0][0].shape[1]  # Calculate the x-coordinate for placing the image
            y = row * images[0][0].shape[0]  # Calculate the y-coordinate for placing the image
            combined_image[y:y + images[0][0].shape[0], x:x + images[0][0].shape[1]] = images[i][0]
            print(f'{i}: {images[i][1]}, location: {row}, {col}')

        # Rotate the image counter-clockwise
        rotated_image = cv2.rotate(combined_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize the rotated image to (400, 200)
        combined_image_rotated = cv2.resize(rotated_image, (int(combined_height/2), int(combined_width/2)))
        # Display or save the combined image
        save_pth = os.path.join(dir_pth, 'stitched.png')
        cv2.imwrite(save_pth, combined_image_rotated)

        cv2.imshow("Combined Image", combined_image_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=='__main__':
    regions = {
    '1-1': [[224.,   7.], [1050.,    5.],[1420., 576.], [49., 578.]],
    '1-2': [[289.,   8.],[993.,   6.],[1356.,  576.], [27., 576.]],
    '1-3': [[288.,   7.],[981.,   7.],[1346.,  577.], [7., 578.]],
    '1-4': [[313.,   8.],[987.,  6.],[1360.,  574.], [12., 578.]],
    '1-5': [[294.,   7.],[961.,   6.],[1407.,  578.], [53., 578.]],
    '1-6': [[249.,  11.],[959.,   8.],[1374.,  577.], [4., 580.]],
    '1-7': [[268.,   6.],[988.,   8.],[1403.,  574.], [28., 578.]],
    '1-8': [[183.,   9.],[956.,   5.],[1409.,  578.], [6., 579.]],
    '1-9': [[260.,   5.],[976.,   4.],[1398.,  578.], [38., 580.]],
    '1-10': [[110.,   6.],[949.,   6.],[1409.,  576.], [4., 578.]],
    '1-11': [[216.,   8.],[944.,   8.],[1368.,    577.], [6., 579.]],
    '1-12': [[300.,   7.],[971.,   8.],[1355.,    576.], [9., 577.]],
    '1-13': [[259.,   7.],[1003.,    7.],[1409.,    577.], [33., 581.]],
    '1-14': [[203.,   9.],[983.,   8.],[1388.,    576.], [6., 580.]],
    '1-15': [[256.,  10.],[956.,   8.],[1377.,    577.], [17., 578.]],
    '1-16': [[234.,   7.],[978.,   5.],[1435.,    579.], [36., 579.]],
    '2-1': [[451.,   8.],[1172.,   11.],[1428.,    577.], [8., 579.]],
    '2-2': [[522.,   3.],[1167.,    6.],[1440.,    577.], [139., 578.]],
    '2-3': [[496.,   8.],[1205.,  10.],[1438.,    579.], [106., 578.]],
    '2-4': [[446.,   6.],[1173.,    6.],[1437.,    578.], [106., 581.]],
    '2-5': [[419.,   5.],[1145.,    7.],[1437.,    580.], [92., 580.]],
    '2-6': [[396.,   7.],[1130.,   10.],[1429.,    578.], [52., 578.]],
    '2-7': [[428. ,  6.],[1123.,    8.],[1437.,    577.], [122., 577.]],
    '2-8': [[391.,   7.],[1061.,   11.],[1371.,    573.], [54.,579.]],
    '2-9': [[506.,   5.],[1149.,    7.],[1438.,    580.], [136., 579.]],
    '2-10': [[479.,   6.],[1134.,    7.],[1419.,    578.], [113., 579.]],
    '2-11': [[392.,   6.],[1095.,    8.],[1432.,    579.], [92., 578.]],
    '2-12': [[449.,   6.],[1108.,    6.],[1425.,    576.] ,[151., 577.]],
    '2-13': [[358.,   7.],[1075.,    8.],[1393.,    573.], [31., 578.]],
    '2-14': [[395.,   7.],[1058.,    6.],[1378.,    578.], [74., 579.]],
    '2-15': [[426.,   5.],[1159.,    8.],[1418.,    578.], [65., 578.]],
    '2-16': [[347.,   8.],[1109.,    6.],[1351.,    574.], [6., 579.]],
    }
    preproc = 1
    main()
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