import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2, os
from scipy.spatial.distance import cdist

def check_if_1s_inside(mask, roi_opp_side_per_cam):
    # Create a polygon mask with the same shape as the input mask
    polygon_mask = np.zeros_like(mask, dtype=np.uint8)
    # Fill the polygon region with ones
    cv2.fillPoly(polygon_mask, [np.asarray(roi_opp_side_per_cam)], 1)
    # Perform bitwise AND operation between the polygon mask and the input mask
    result = cv2.bitwise_and(mask, polygon_mask)
    # Check if there are any non-zero (1) values in the result
    contains_ones = np.any(result)
    return result
# def poi_in_polygon(roi, center):
#     # Create a mask with the selected polygon
#     is_inside = cv2.pointPolygonTest(np.array(roi), center, False)
#     return is_inside
def check_bins_status(img_dict, oppo_side_region, yellow_pts_thres, voting_ratio_thr):
    '''
    input: images for the whole cams, at the same time moment / region dictionaries for the whole cams
           / points threshold (same for every cam) / how many cams vote for bins-down threshold
    output: final decision for bins-down that all cams say.
    note: if a cam has no oppo region, then just move to the next cam
    '''
    bins_down_total = {}
    for cam_num, roi_opp_side in oppo_side_region.items():
        if len(roi_opp_side) < 2:
            continue
        extracted_mask = extract_food_bins(img_dict[cam_num])
        result = check_if_1s_inside(extracted_mask, roi_opp_side)
        how_many_ones = np.sum(result)
        print(f'{cam_num} yellow size: {how_many_ones}')
        if how_many_ones > yellow_pts_thres:
            bins_down = 1 # 1 means 'down'
        else:
            bins_down = 0 # 0 means 'up'
        bins_down_total[cam_num] = bins_down
    voting_ratio = sum(bins_down_total.values())/len(bins_down_total.values())
    print(f'Cameras voting percentage for bins-down: {voting_ratio:.2f}')
    if voting_ratio >= voting_ratio_thr:
        status = 'down'  # 1
    else:
        status = 'up'   # 0
        voting_ratio = 1.0 - voting_ratio
    return status, voting_ratio

def distribution_calculation(anchors, centers):  # gini coefficient, iqr

    concentration_level = 0.0
    return concentration_level

def roi_concentration_calculation(centers, rois):  # food bin region concentration

    roi_concentration_level = 0.0
    return roi_concentration_level

def feeding_stage():

    stage = 'early' # 'early', 'mid', 'last'
    return stage

def feeding_time(bins_down_time, feeding_last_stage_time):

    eating_mins =  feeding_last_stage_time - bins_down_time
    return eating_mins

# def show_anns(anns, fn, size_limit_Lg_num, size_limit_sm_num,
#     # ======================================================================================
#     # Write text on the image
#     # text = f'Density level: \n{den_in_rois:.4f}\nFeeding: {feeding}'
#     cam = fn.split('/')[-1].split('_')[0] + '_' + fn.split('_')[1]
#     # text = f'Distribution level: \nmean: {mean:.4f}\nstd: {std:.4f}\ncam: {cam}\nIQR: {iqr}' \
#     #        f'\nginico: {ginico:.4f}'
#     text = f'Concentration level: \nIQR: {iqr}\nginico: {ginico:.4f}\ncam: {cam}\n' \
#            f'bins: {bins_up}\nFeeding: {feeding}'
#     x = 50  # x-coordinate of the text position
#     y = 300  # y-coordinate of the text position
#     fontsize = 25
#     fontcolor = 'red'
#     fontweight = 'bold'
#     ax.text(x, y, text, fontsize=fontsize, color=fontcolor, weight=fontweight)
#     ax.imshow(img)
#     # =========================================================================================
def poi_in_polygon(roi, center, image):
    # Create a mask with the selected polygon
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [np.array(roi)], 255)
    is_inside = cv2.pointPolygonTest(np.array(roi), center, False)
    return is_inside
def how_many_chks_in_poly(roi, centers, image):
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [np.array(roi)], 255)
    cnt = 0
    for center in centers:
        is_inside = cv2.pointPolygonTest(np.array(roi), center, False)
        if is_inside > 0: cnt += 1
    return cnt
def create_grid(image_shape, d):
    height, width = image_shape[:2]
    x = np.arange(0, width, d)
    y = np.arange(0, height, d)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx, yy), axis=2)
    return grid
def count_points_within_distance(grid, points, distance):
    distances = cdist(grid.reshape(-1, 2), points.reshape(-1, 2))
    counts = np.sum(distances <= distance, axis=1)
    return counts
def make_heatmap_a_number(counts):
    cnt_wo_zeros = counts[counts != 0]
    mean = np.mean(cnt_wo_zeros)
    std = np.std(cnt_wo_zeros)
    # Calculate the 25th and 75th percentiles
    q25 = np.percentile(counts, 25)
    q75 = np.percentile(counts, 75)
    # Calculate the IQR
    iqr = q75 - q25
    return mean, std, iqr
def calculate_ginicoeffi(cam_num, counts):
    # Sort the data in ascending order
    sorted_data = np.sort(counts)
    # Calculate the cumulative population fraction
    cumulative_pop = np.cumsum(sorted_data) / np.sum(sorted_data)
    # Calculate the Lorenz curve values
    lorenz_curve = np.concatenate(([0], cumulative_pop))
    # Calculate the Gini coefficient
    gini_coefficient = 1 - np.sum(lorenz_curve[:-1] + lorenz_curve[1:]) / len(lorenz_curve)
    lower_bound = 0.3
    upper_bound = 0.6
    clipped_value = max(lower_bound, min(gini_coefficient, upper_bound))
    scaled_ginico = (clipped_value - lower_bound) / (upper_bound - lower_bound)
    print(f"Gini Coefficient for {cam_num}: {scaled_ginico:.4f}, {gini_coefficient:.4f}")
    return scaled_ginico
def extract_food_bins(img):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_yellow = np.array([15, 100, 100])
    # upper_yellow = np.array([34, 255, 255])
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask_yellow
def compose_whole_image(image_dict_per_cam, dir_pth, roi_stitching_region,
                        bins_down, bins_down_confidence_score, overall_ginico, time_stamp,
                        show_img = True):
    image_dict_per_cam_1 = {}
    for i in list(image_dict_per_cam.keys()):
        if len(i) == 3:
            image_dict_per_cam_1[i.replace('-', '-0')] = image_dict_per_cam[i]
        else:
            image_dict_per_cam_1[i] = image_dict_per_cam[i]
    image_dict_per_cam_sorted = dict(sorted(image_dict_per_cam_1.items()))
    # images = list(image_dict_per_cam_sorted.values())
    # Create a blank canvas to hold the combined image
    output_width = 500
    output_height = 400
    combined_height = output_height * 16  # Height of the combined image
    combined_width = output_width * 2  # Width of the combined image
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    # Iterate over the images and place them in the combined image
    for cam_num in image_dict_per_cam_sorted.keys():
        image = image_dict_per_cam_sorted[cam_num]
        image = image_stretching_to_rectangle(image, roi_stitching_region[cam_num.replace('-0', '-')],
                                              output_width, output_height)
        xx = int(cam_num.split('-')[0])
        yy = int(cam_num.split('-')[1])
        if xx == 1:
            row = yy - 1
            col = xx - 1
        else:
            row = 15 - (yy - 1)
            col = xx - 1
        x = col * image.shape[1]  # Calculate the x-coordinate for placing the image
        y = row * image.shape[0]  # Calculate the y-coordinate for placing the image
        combined_image[y:y + image.shape[0], x:x + image.shape[1]] = image
        # print(f'{cam_num} location: {row}, {col}')
    # Rotate the image counter-clockwise
    rotated_image = cv2.rotate(combined_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Resize the rotated image to (400, 200)
    combined_image_rotated = cv2.resize(rotated_image, (int(combined_height / 4), int(combined_width / 4)))
    # Display or save the combined image
    # Define the text to write
    if bins_down == 'down':
        text = f'{time_stamp}/ food_bins: {bins_down} / Confidence: {bins_down_confidence_score}' \
               f'food_concentration: {overall_ginico:.2f} / '
    elif bins_down == 'up':
        text = f'{time_stamp}/ food_bins: {bins_down} / Confidence: {bins_down_confidence_score}' \
               f'/ distribution_level: {overall_ginico:.2f} / '
    # Choose the font type, size, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_color = (0, 0, 255)  # White color in BGR format
    # Choose the position to write the text
    position = (50, 50)  # (x, y) coordinates
    # Write the text on the image
    cv2.putText(combined_image_rotated, text, position, font, font_scale, font_color, thickness=2)

    save_pth = os.path.join(dir_pth, f'{time_stamp}_stitched.png')
    if not os.path.exists(dir_pth):
        os.makedirs(dir_pth)
    cv2.imwrite(save_pth, combined_image_rotated)
    if show_img == True:
        cv2.imshow("Combined Image", combined_image_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def image_stretching_to_rectangle(image, stretching_region, output_width, output_height):
    points = np.asarray(stretching_region, dtype=np.float32)
    # Define the desired width and height of the output rectangle
    # Define the 4 corner points of the output rectangle in clockwise order starting from top-left
    output_points = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32)
    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(points, output_points)
    # Apply the perspective transformation to the image
    result = cv2.warpPerspective(image, matrix, (output_width, output_height))
    # cv2.imshow("Transformed Image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result
def get_overall_ginico(ginicos_dict_per_cam):
    raw_mean = np.mean(np.asarray(list(ginicos_dict_per_cam.values())))
    # lower_bound = 0.3
    # upper_bound = 0.7
    # clipped_value = max(lower_bound, min(raw_mean, upper_bound))
    # scaled_mean = (clipped_value - lower_bound) / (upper_bound - lower_bound)
    return raw_mean

