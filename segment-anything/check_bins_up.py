import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from extract_food_bins import extract_food_bins

# fn = 'data/150005_2-11_etc.jpg'
# fn = 'data/172925_2-13_4749540_2.jpg'
# fn = 'data/172925_1-10_7387700_2.jpg'
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
fn = 'data/172925_2-13_4749540_2.jpg'
# fn = 'data/bins_only/140006_2-11_etc_bins_only_woLegend.png'
# fn = 'data/bins_only/161525_2-11_feeding_bins_only_woLegend.png'
# fn = 'data/bins_only/161525_2-11_feeding_bins_only_woLegend.png'
# fn = 'results/heatmap.jpg'

size_limit_Lg_num = 0
size_limit_sm_num = 700
centers_in_roi = []
den_in_rois = 0
counts = 0
roi = [(422, 5), (1093, 12), (1423, 581), (164, 581)]
roi_0_1 = [(503, 6), (696, 4), (680, 577), (335, 579)]
roi_2 = [(730, 7), (873, 6), (986, 579), (725, 578)]
roi_3 = [(878, 3), (1003, 7), (1129, 272), (967, 274)]
# roi_opp_side = [(6, 9), (206, 9), (12, 253)]
roi_opp_side = [(49, 9), (4, 66), (24, 163), (142, 10)]
bins_up = None
yellow_pts_thres = 50
def show_anns( save_pth = None):
    global bins_up

    extracted_mask = extract_food_bins(fn)
    # plt.imshow(extracted_mask)
    # plt.show()
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
    # fn_base = fn.split('/')[-1].split('.')[0]
    # save_pth = f'results/{fn_base}_bins_{bins_up}.png'
    # plt.savefig(save_pth)

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

    # Print the result
    # print("Polygon contains ones:", contains_ones)
image = cv2.imread(fn)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../weight/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# model=sam
# points_per_side=32
# pred_iou_thresh=0.80
# stability_score_thresh=0.80
# crop_n_layers=1
# crop_n_points_downscale_factor=2
# min_mask_region_area=100

# mask_generator_2 = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=points_per_side,
#     pred_iou_thresh=pred_iou_thresh,
#     stability_score_thresh=stability_score_thresh,
#     crop_n_layers=crop_n_layers,
#     crop_n_points_downscale_factor=crop_n_points_downscale_factor,
#     min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
# )
# masks2 = mask_generator_2.generate(image)
#
# print(len(masks2))

save_fig_name_density_base = './results/' + fn.split('/')[-1].split('.')[0] + '_binsUpCheck'
fig = plt.figure(figsize=(20,20))

plt.imshow(image)
plt.show()
show_anns(save_pth = save_fig_name_density_base)


# plt.axis('off')
# plt.show()
