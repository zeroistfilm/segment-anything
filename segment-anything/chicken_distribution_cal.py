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
import cv2, time
from scipy.spatial.distance import cdist
from extract_food_bins import extract_food_bins
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from matplotlib.backend_bases import MouseButton
'''
This is the code specified for cam 2-11 setting. 
Parameter setting information is required to generalize to all the cams. 
'''
# fn = 'data/110008_2-11_etc.jpg'
# fn = 'data/155006_2-11_feeding.jpg'
# fn = 'data/135006_2-11_etc.jpg'
# fn = 'data/135526_2-11_etc_0.jpg'
# fn = 'data/135805_2-11_4216416_2.jpg'
# fn = 'data/140006_2-11_etc.jpg'
# fn = 'data/160005_2-11_med_feeding.jpg'
# fn = 'data/161005_2-11_med_feeding_1.jpg'
# fn = 'data/161525_2-11_feeding.jpg'
fn = 'data/162009_2-11_later_feeding.jpg'
fn_1 = 'data/172925_2-13_4749540_2.jpg'


size_limit_Lg_num = 8
size_limit_sm_num = 700
centers_in_roi = []
den_in_rois = 0
counts = 0
roi = [(422, 5), (1093, 12), (1423, 581), (164, 581)]
roi_0_1 = [(503, 6), (696, 4), (680, 577), (335, 579)]
roi_2 = [(730, 7), (873, 6), (986, 579), (725, 578)]
roi_3 = [(878, 3), (1003, 7), (1129, 272), (967, 274)]
roi_opp_side = [(49, 9), (4, 66), (24, 163), (142, 10)]
bins_up = None
yellow_pts_thres = 50
text = None
iqr, ginico = None, None
def show_anns(anns, save_pth = None):
    global  centers_in_roi, den_in_rois, counts, text, iqr, ginico, bins_up
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']),
                         reverse=True)[size_limit_Lg_num:size_limit_sm_num]
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # ================================ check if bins up or down ==============================
    extracted_mask = extract_food_bins(fn)
    result = check_if_1s_inside(extracted_mask, roi_opp_side)
    # Check if there are any non-zero (1) values in the result
    # contains_ones = np.any(result)
    how_many_ones = np.sum(result)
    print(f'yellow size: {how_many_ones}')
    if how_many_ones > yellow_pts_thres:
        bins_up = 'down'
        print('Food bins down')
    else:
        bins_up = 'up'
        print('Food bins up')
    # ========================= mask preparation ===================================
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    # ===========================================================================================================
    for ann in sorted_anns:
        # m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask
        bb = ann['bbox']
        center = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        if poi_in_polygon(roi, center)>0:
            centers_in_roi.append(center)
            # plt.plot(center[0], center[1], marker='v', color="red")
    centers_in_roi_np = np.asarray(centers_in_roi)
    # ======== crete and plot grids =============
    grid = create_grid(image.shape, 15)
    # plt.scatter(grid[:, :, 0], grid[:, :, 1], color='green', s=10)
    # Define the distance threshold
    distance_threshold = 100
    # ========= Count points within distance for each grid point ==============
    counts = count_points_within_distance(grid, centers_in_roi_np, distance_threshold)
    # ========= Plot the counts as text ==============
    # for i, count in enumerate(counts):
    #     x, y = grid[i // grid.shape[1], i % grid.shape[1]]
    #     plt.text(x, y, str(count), color='pink', fontsize=8)
    # ================ Create the heatmap =========================
    # Add colorbar with custom colormap
    # heatmap = plt.imshow(counts.reshape((-1,grid.shape[1])), cmap='cool', alpha=0.5, extent=(0, image.shape[1], image.shape[0], 0))
    heatmap = plt.imshow(counts.reshape((-1, grid.shape[1])), cmap='hot', alpha=0.5,
                         extent=(0, image.shape[1], image.shape[0], 0))

    colorbar = plt.colorbar(heatmap, label='Count', cmap='hot')
    # ================ express the heatmap as a number =========================
    mean, std, iqr = make_heatmap_a_number(counts)
    ginico = calculate_ginicoeffi(counts)
    # =======================================================================
    # cnt_0_1_num = how_many_chks_in_poly(roi_0_1, centers_in_roi)
    # cnt_2_num = how_many_chks_in_poly(roi_2, centers_in_roi)
    # cnt_3_num = how_many_chks_in_poly(roi_3, centers_in_roi)
    # tot_in_rois = cnt_0_1_num + cnt_2_num + cnt_3_num
    # num_out_rois = len(centers_in_roi) - tot_in_rois
    # den_in_rois = tot_in_rois/(tot_in_rois + num_out_rois * wgt_outside_roi)
    print(f'density around feeding ball: {den_in_rois}')
    # ========================= Determine if feeding or not ================================
    # if den_in_rois>= 0.5:
    #     feeding = True
    # else: feeding = False
    if bins_up == 'down': feeding = True
    elif bins_up == 'up': feeding = False
    # ======================================================================================
    # Write text on the image
    # text = f'Density level: \n{den_in_rois:.4f}\nFeeding: {feeding}'
    cam = fn.split('/')[-1].split('_')[0] + '_' + fn.split('_')[1]
    # text = f'Distribution level: \nmean: {mean:.4f}\nstd: {std:.4f}\ncam: {cam}\nIQR: {iqr}' \
    #        f'\nginico: {ginico:.4f}'
    text = f'Concentration level: \nIQR: {iqr}\nginico: {ginico:.4f}\ncam: {cam}\n' \
           f'bins: {bins_up}\nFeeding: {feeding}'
    x = 50  # x-coordinate of the text position
    y = 300  # y-coordinate of the text position
    fontsize = 25
    fontcolor = 'red'
    fontweight = 'bold'
    ax.text(x, y, text, fontsize=fontsize, color=fontcolor, weight=fontweight)
    ax.imshow(img)
    # ==========================================================================================================
    # plt.plot(center[0], center[1], marker='v', color="white")
    save_pth = save_pth + f'_IQR_{iqr:.3f}_ginico_{ginico:.3f}_bins_{bins_up}.png'
    plt.savefig(save_pth)

def poi_in_polygon(roi, center):
    # Create a mask with the selected polygon
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [np.array(roi)], 255)
    is_inside = cv2.pointPolygonTest(np.array(roi), center, False)
    return is_inside
def how_many_chks_in_poly(roi, centers):
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
def calculate_ginicoeffi(counts):
    # Sort the data in ascending order
    sorted_data = np.sort(counts)

    # Calculate the cumulative population fraction
    cumulative_pop = np.cumsum(sorted_data) / np.sum(sorted_data)

    # Calculate the Lorenz curve values
    lorenz_curve = np.concatenate(([0], cumulative_pop))

    # Calculate the Gini coefficient
    gini_coefficient = 1 - np.sum(lorenz_curve[:-1] + lorenz_curve[1:]) / len(lorenz_curve)

    print("Gini Coefficient:", gini_coefficient)
    return gini_coefficient
def check_if_1s_inside(mask, poly):
    # Create a polygon mask with the same shape as the input mask
    polygon_mask = np.zeros_like(mask, dtype=np.uint8)

    # Fill the polygon region with ones
    cv2.fillPoly(polygon_mask, [np.asarray(poly)], 1)

    # Perform bitwise AND operation between the polygon mask and the input mask
    result = cv2.bitwise_and(mask, polygon_mask)

    # Check if there are any non-zero (1) values in the result
    contains_ones = np.any(result)
    return result
image = cv2.imread(fn)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grid = create_grid(image.shape, 100)
wgt_outside_roi = 2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../weight/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

points_per_side=64
pred_iou_thresh=0.50
stability_score_thresh=0.80
crop_n_layers=1
crop_n_points_downscale_factor=2
min_mask_region_area=100

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=points_per_side,
    pred_iou_thresh=pred_iou_thresh,
    stability_score_thresh=stability_score_thresh,
    crop_n_layers=crop_n_layers,
    crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
)
start = time.time()
# for image in images:
masks2 = mask_generator_2.generate(image)
print(f'elapsed time: {time.time() - start}')
start = time.time()
areas = [x['area'] for x in masks2]

print(len(masks2))

save_fig_name_density_base = './results/analysis/' + fn.split('/')[-1].split('.')[0] + '_Heatmap'
fig = plt.figure(figsize=(20,20))

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
# plt.show()
show_anns(masks2, save_pth = save_fig_name_density_base)
# ============================== save Analysis result on original image =========================
fig1 = plt.figure(figsize=(20,20))
plt.imshow(image_rgb)
x = 50  # x-coordinate of the text position
y = 250  # y-coordinate of the text position
fontsize = 25
fontcolor = 'red'
fontweight = 'bold'
ax = plt.gca()
ax.set_autoscale_on(False)
ax.text(x, y, text, fontsize=fontsize, color=fontcolor, weight=fontweight)
ax.imshow(image_rgb)

save_fig_name_base = './results/analysis/' + fn.split('/')[-1].split('.')[0] + '_Analysis_'

save_pth = save_fig_name_base + f'_IQR_{iqr:.3f}_ginico_{ginico:.3f}_bins_{bins_up}.png'
plt.savefig(save_pth)

plt.axis('off')
plt.show()

