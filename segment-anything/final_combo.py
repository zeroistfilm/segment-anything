'''
coverage:
1. Each cam -> distribution level calculatioin
2. bins up or down -> feeding start / end
                   -> per line determination
3. Feeding level   -> each food bin concentration level -> broken or not
                   -> Early - Med - later - very last
4. Feeding time    -> Very last - Early
5. Image combination
5. Disply above information on Combo Image
'''
import cv2, os, glob, json, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def imgs_masks_jsons_specific_time(date, time_stamp, write):
    # ======================================== import images ==============================================
    # Load the image
    # dir_pth = 'data/100005_data'
    # dir_pth = 'data/140039_data'
    # time_stamp = ''  # 090036 091004 090517 085522 082636
    dir_pth = f'data/{date}/{time_stamp}_data'
    directory = f"{dir_pth.split('_')[0]}_images_masks_jsons"
    if not os.path.exists(directory) and write == True:
        os.makedirs(directory)
    imgs_save_to = f"{directory}/{dir_pth.split('_')[0].split('/')[-1]}_images_cv2read.json"
    masks_save_to = f"{directory}/{dir_pth.split('_')[0].split('/')[-1]}_masksPerImage.json"
    fns = glob.glob(os.path.join(dir_pth, '*.jpg'))
    whole_img_dict = {} # key: image cam-num, value: cv2 image values
    masks_per_camNum = {}
    for fn in fns:
        image = cv2.imread(fn)
        fn_base = fn.split('_')[2]
        whole_img_dict[fn_base] = image
    # ======================================== chicken centers SAM ==============================================
    sam_checkpoint = "../weight/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    points_per_side = 32
    pred_iou_thresh = 0.50
    stability_score_thresh = 0.80
    crop_n_layers = 1
    crop_n_points_downscale_factor = 2
    min_mask_region_area = 100
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
    start_whole = time.time()
    for i, cam_num in enumerate(whole_img_dict.keys()):
        image = whole_img_dict[cam_num]
        masks2 = mask_generator_2.generate(image)
        masks_per_camNum[cam_num] = masks2
        print(f'{cam_num} mask elapsed: {time.time() - start}')
        start = time.time()
    print(f'Elapsed for the whole analysis: {time.time() - start_whole}')
    if write == True:
        with open(imgs_save_to, "wb") as file:
            pickle.dump(whole_img_dict, file)
        with open(masks_save_to, "wb") as file:
            pickle.dump(masks_per_camNum, file)
    return whole_img_dict, masks_per_camNum
    # ================================= Distribution calculation ==========================================

    # ========================================== bins up or down ==========================================

    # ======================================== Feeding level ==============================================


# if __name__=='__main__':
#
#     preproc = 1
#     main()
