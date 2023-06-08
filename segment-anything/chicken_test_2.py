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
import cv2
from scipy.spatial import cKDTree

from matplotlib.backend_bases import MouseButton

size_limit_Lg_num = 8
size_limit_sm_num = 700
centers_in_roi = []
den_in_rois = 0

# fn = 'data/110008_2-11_etc.jpg'
# fn = 'data/155006_2-11_feeding.jpg'
# fn = 'data/135006_2-11_etc.jpg'
# fn = 'data/135526_2-11_etc_0.jpg'
# fn = 'data/135805_2-11_4216416_2.jpg'
# fn = 'data/140006_2-11_etc.jpg'
# fn = 'data/160005_2-11_med_feeding.jpg'
# fn = 'data/161005_2-11_med_feeding_1.jpg'
# fn = 'data/161525_2-11_feeding.jpg'
# fn = 'data/162009_2-11_later_feeding.jpg'
# fn = 'data/172925_2-13_4749540_2.jpg'
fn = 'data/100005_data/100005_1-2_5607888_1.jpg'
def show_anns(anns, save_pth = None):
    global  centers_in_roi, den_in_rois, bins_up
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']),
                         reverse=True)[size_limit_Lg_num:size_limit_sm_num]
    ax = plt.gca()
    ax.set_autoscale_on(False)
    bb = sorted_anns[30]['bbox']
    print(bb)
    x0, y0 = bb[0], bb[1]
    w, h = bb[2] , bb[3]

    # center = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
    # ax.add_patch(patches.Rectangle((x0, y0),w,h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask
        bb = ann['bbox']
        center = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        if poi_in_polygon(roi, center)>0:
            centers_in_roi.append(center)
            plt.plot(center[0], center[1], marker='v', color="red")
    cnt_0_1_num = how_many_chks_in_poly(roi_0_1, centers_in_roi)
    cnt_2_num = how_many_chks_in_poly(roi_2, centers_in_roi)
    cnt_3_num = how_many_chks_in_poly(roi_3, centers_in_roi)
    tot_in_rois = cnt_0_1_num + cnt_2_num + cnt_3_num
    num_out_rois = len(centers_in_roi) - tot_in_rois
    den_in_rois = tot_in_rois/(tot_in_rois + num_out_rois * wgt_outside_roi)
    print(f'density around feeding ball: {den_in_rois}')

    # m = sorted_anns[30]['segmentation']
    # color_mask = np.concatenate([np.random.random(3), [0.35]])
    # img[m] = color_mask
    # Write text on the image
    if den_in_rois>= 0.5:
        feeding = True
    else: feeding = False
    text = f'Density level: \n{den_in_rois:.4f}\nFeeding: {feeding}'
    x = 50  # x-coordinate of the text position
    y = 150  # y-coordinate of the text position
    fontsize = 25
    fontcolor = 'red'
    fontweight = 'bold'
    ax.text(x, y, text, fontsize=fontsize, color=fontcolor, weight=fontweight)

    ax.imshow(img)
    # plt.plot(center[0], center[1], marker='v', color="white")
    save_pth = save_pth + f'_density_{den_in_rois:.4f}.png'
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

# image = cv2.imread('data/155006_2-11_feeding.jpg')  # 0.7325
# image = cv2.imread('data/135526_2-11_etc_0.jpg')  # density: 0.4
# image = cv2.imread('data/160005_2-11_med_feeding.jpg')  # 0.6370860927152318
# image = cv2.imread('data/161005_2-11_med_feeding_1.jpg')  # 0.5
# image = cv2.imread('data/162009_2-11_later_feeding.jpg')  # 0.4730
# image = cv2.imread('data/150005_2-11_etc.jpg')  # 0.5
image = cv2.imread(fn)  # 0.5
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
roi = [(422, 5), (1093, 12), (1423, 581), (164, 581)]
roi_0_1 = [(503, 6), (696, 4), (680, 577), (335, 579)]
roi_2 = [(730, 7), (873, 6), (986, 579), (725, 578)]
roi_3 = [(878, 3), (1003, 7), (1129, 272), (967, 274)]
wgt_outside_roi = 2
# print(poi_in_polygon(roi, (1,1)))
# 172925_2-13_4749540_2
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../weight/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)
# print(len(masks))
# print(masks[0].keys())
#
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()
model=sam
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
masks2 = mask_generator_2.generate(image)
areas = [x['area'] for x in masks2]
HIST_BINS = np.linspace(0, 1000, 100)
counts, bins = np.histogram(areas, HIST_BINS)

# plt.stairs(counts, bins)
print(len(masks2))
save_fig_name = f'./results/pps{points_per_side}_pit' \
                f'{pred_iou_thresh}_sst{stability_score_thresh}' \
                f'_cnl{crop_n_layers}_cnpdf{crop_n_points_downscale_factor}' \
                f'_mmra{min_mask_region_area}_sizeSM{size_limit_sm_num}' \
                f'_sizeLG{size_limit_Lg_num}.png'
save_fig_name_density_base = './results/' + fn.split('/')[-1].split('.')[0] + '_noshow'
fig = plt.figure(figsize=(20,20))

plt.imshow(image)
show_anns(masks2, save_pth = save_fig_name_density_base)

plt.axis('off')
plt.show()

sorted_anns = sorted(masks2, key=(lambda x: x['area']),
                     reverse=True)[size_limit_Lg_num:size_limit_sm_num]
ax = plt.gca()
ax.set_autoscale_on(False)

# -------------------------------------------------------------------------
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Find the coordinates of the points
# points = np.argwhere(binary > 0)
points = np.asarray(centers_in_roi)

# Calculate the distance between each point and its nearest neighbor
tree = cKDTree(points)
distances, _ = tree.query(points, k=2)

# Calculate the average nearest neighbor distance
average_distance = np.mean(distances[:, 1])
print(average_distance)
# Determine if the points are evenly distributed or concentrated
if average_distance <= 1.0:
    print("Points are concentrated")
else:
    print("Points are evenly distributed")

'''
135526_2-11_etc_0.jpg:   average_distance = 14.03936

'''