import cv2
import numpy as np
import glob, os
import json
# Global variables
points = []

def select_region(fns, save_fn):
    global points
    def on_mouse(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow(f'{cam_num}-{i}/{len(fns)}', image)
    roi_dict = {}
    for i, fn in enumerate(fns):
        points = []
        image = cv2.imread(fn)
        cam_num = fn.split('_')[2]
        clone = image.copy()
        # Create a window and display the image
        cv2.namedWindow(f'{cam_num}-{i}/{len(fns)}')
        cv2.imshow(f'{cam_num}-{i}/{len(fns)}', image)
        # Set the mouse callback function
        cv2.setMouseCallback(f'{cam_num}-{i}/{len(fns)}', on_mouse)
        # Wait for the user to select points
        cv2.waitKey(0)
        # Close the window
        cv2.destroyAllWindows()
        roi_dict[cam_num] = points
        print(f'{cam_num}: {points}')
    # Save the dictionary to a file
    print(roi_dict)
    with open(f"parameters/{save_fn}.json", "w") as file:
        json.dump(roi_dict, file)
def select_region_bins_down(fns, save_fn):
    global points
    def on_mouse(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow(f'{cam_num}-{i}/{len(fns)}', image)
    roi_dict = {}
    for i, fn in enumerate(fns):
        points = []
        image = cv2.imread(fn)
        cam_num = fn.split('_')[-1].split('.')[0]
        if cam_num in bins_check_cam_list:
            # Create a window and display the image
            cv2.namedWindow(f'{cam_num}-{i}/{len(fns)}')
            cv2.imshow(f'{cam_num}-{i}/{len(fns)}', image)
            # Set the mouse callback function
            cv2.setMouseCallback(f'{cam_num}-{i}/{len(fns)}', on_mouse)
            # Wait for the user to select points
            cv2.waitKey(0)
            # Close the window
            cv2.destroyAllWindows()
            roi_dict[cam_num] = points
            print(f'{cam_num}: {points}')
        else:
            continue
    # Save the dictionary to a file
    print(roi_dict)
    with open(f"parameters/{save_fn}.json", "w") as file:
        json.dump(roi_dict, file)
def roi_check_bins_down(param, fns):
    # Create a mask with the selected polygon
    f = open(param)
    # returns JSON object as a dictionary
    params = json.load(f)
    for i, fn in enumerate(fns):
        image = cv2.imread(fn)
        # cam_num = fn.split('_')[2]
        cam_num = fn.split('_')[2]
        if cam_num in bins_check_cam_list:
            points = params[cam_num]
            # clone = image.copy()
            # mask = np.zeros_like(image[:, :, 0])
            print(points)
            # cv2.fillPoly(image, [np.array(points)], 255)
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            # Create a window and display the image
            cv2.namedWindow(cam_num)
            cv2.imshow(cam_num, image)

            # Wait for the user to select points
            cv2.waitKey(0)
            # Close the window
            cv2.destroyAllWindows()
def roi_check(param, fns):
    # Create a mask with the selected polygon
    f = open(param)
    # returns JSON object as a dictionary
    params = json.load(f)
    for i, fn in enumerate(fns):
        image = cv2.imread(fn)
        # cam_num = fn.split('_')[2]
        cam_num = fn.split('_')[-1].split('.')[0]
        points = params[cam_num]
        # clone = image.copy()
        # mask = np.zeros_like(image[:, :, 0])
        print(points)
        # cv2.fillPoly(image, [np.array(points)], 255)
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        # Create a window and display the image
        cv2.namedWindow("Image")
        cv2.imshow("Image", image)

        # Wait for the user to select points
        cv2.waitKey(0)
        # Close the window
        cv2.destroyAllWindows()

def main():
    if preproc != 7:
        select_region(fns, regions[preproc])
    elif preproc == 7:
        select_region_bins_down(fns, regions[preproc])
if __name__=='__main__':
    preproc = 7

    # Load the image
    # dir_pth = 'data/100005_data'
    # dir_pth = 'results/100005_food_bins'
    dir_pth = 'data/140039_data'
    fns = glob.glob(os.path.join(dir_pth, '*.jpg'))
    regions = {0: 'roi_foodbin_slate_0_1',
               1: 'roi_foodbin_ground_2',
               2: 'roi_foodbin_ground_3',
               3: 'roi_foodbin_wall',
               4: 'roi_drink',
               5: 'roi_chkn_cnt_region',
               6: 'roi_stitching_region',
               7: 'roi_bins_down_region'}
    bins_check_cam_list = ['1-3', '1-5','1-11','1-12','1-16',
                           '2-2','2-3','2-4','2-5','2-7','2-9','2-10' ,'2-11','2-12','2-14','2-15']
    roi_foodbin_slate_0_1 = {}
    roi_foodbin_ground_2 = {}  # ground - slate side bins
    roi_foodbin_ground_3 = {}  # ground - wall side bins
    roi_foodbin_wall = {}  # packed to wall
    roi_drink = {}
    roi_chkn_cnt_region = {}
    # main()
    if preproc == 7:
        roi_check_bins_down(f'parameters/{regions[preproc]}.json', fns)
    else: roi_check(f'parameters/{regions[preproc]}.json', fns)
    '''
    preproc = 0 :  roi_foodbin_slate_0_1
    preproc = 1 :  roi_foodbin_ground_2
    preproc = 2 :  roi_foodbin_ground_3
    preproc = 3 :  roi_foodbin_wall
    preproc = 4 :  roi_drink

    '''