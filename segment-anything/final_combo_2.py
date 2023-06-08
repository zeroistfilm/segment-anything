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
import cv2, os, glob, json, time, pickle, csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from functions.utils_uniai import check_bins_status, poi_in_polygon, \
    create_grid, count_points_within_distance, make_heatmap_a_number, \
    calculate_ginicoeffi, compose_whole_image, get_overall_ginico
from final_combo import imgs_masks_jsons_specific_time
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def main():
    '''
    time_stamp:
    082636 082730 083001 084001 085000 085522 090036 090517 091004 091518 092000 092946
    100005
    140039   160037
    '''
    # time_stamps = ['082636', '082730', '083001', '084001', '085000', '085522', '090036',
    #                '090517', '091004', '091518', '092000', '092946',
    #                '094039', '095000', '100038',  '101023', '102000', '103036']
    # time_stamps = ['104041', '105040', '110002', '111000', '112004', '113041', ]
    # time_stamps = ['114002', '115037', '120002', '121037', ]
    date = '230530'
    direc_pth = f'data/{date}'
    time_stamps = [name.split('_')[0] for name in os.listdir(direc_pth) if os.path.isdir(os.path.join(direc_pth, name))]
    time_stamps = sorted(time_stamps)
    analysis = []
    write = False
    for time_stamp in time_stamps:
        print(f'=============== Analyzing time: {date}-{time_stamp} ==================')
        whole_img_dict, masks_per_camNum = imgs_masks_jsons_specific_time(date, time_stamp, write=write)
        # ======================================== import images ==============================================
        # Load the images from all cameras and masks from each image
        if write == False:
            img_masks_json_dir_pth = f'data/{time_stamp}_images_masks_jsons'
            with open(f"{img_masks_json_dir_pth}/{time_stamp}_images_cv2read.json", "rb") as file:
                whole_img_dict = pickle.load(file)  # key: image cam-num, value: cv2 image values
            with open(f"{img_masks_json_dir_pth}/{time_stamp}_masksPerImage.json", "rb") as file:
                masks_per_camNum = pickle.load(file)  # key: cam-num, value: masks for each image
        # time_stamp = img_masks_json_dir_pth.split('/')[-1].split('_')[0]
        # ======================================== chicken centers SAM ==============================================
        ginicos_dict_per_cam = {}
        iqrs_dict_per_cam = {}
        for cam_num, masks in masks_per_camNum.items():
            # ============= for one camera ================
            centers_in_roi = []
            sorted_anns = sorted(masks, key=(lambda x: x['area']),
                                 reverse=True)[size_limit_Lg_num:size_limit_sm_num]
            for ann in sorted_anns:
                # ============= for each mask of the camera ================
                bb = ann['bbox']
                center = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
                if poi_in_polygon(roi_chkn_cnt_region[cam_num], center, whole_img_dict[cam_num]) > 0:
                    centers_in_roi.append(center)
            if len(centers_in_roi) < 3:
                continue
            centers_in_roi_np = np.asarray(centers_in_roi)
        # ================================= Distribution calculation ==========================================
            # ======== crete and plot grids =============
            grid = create_grid(whole_img_dict[cam_num].shape, 15)
            # plt.scatter(grid[:, :, 0], grid[:, :, 1], color='green', s=10)
            # Define the distance threshold
            distance_threshold = 100
            # ========= Count points within distance for each grid point ==============
            counts_per_cam = count_points_within_distance(grid, centers_in_roi_np, distance_threshold)
            # ================ express the heatmap as iqr and ginico =========================
            mean, std, iqr = make_heatmap_a_number(counts_per_cam)
            ginico = calculate_ginicoeffi(cam_num, counts_per_cam)
            ginicos_dict_per_cam[cam_num] = ginico
            iqrs_dict_per_cam[cam_num] = iqr
        overall_ginico = get_overall_ginico(ginicos_dict_per_cam)
        overall_iqr = np.mean(np.asarray(list(iqrs_dict_per_cam.values())))
        # ========================================== bins up or down ==========================================
        bins_down, bins_down_confidence_score = check_bins_status(img_dict=whole_img_dict,
                                      oppo_side_region=roi_bins_down_region,
                                      yellow_pts_thres=yellow_pts_thres,
                                      voting_ratio_thr = voting_ratio_thr)
        # ======================================== Feeding level ==============================================
        if bins_down == 'down':
            print(f'Current bins status: {bins_down}')
            print(f'Overall ginico is: {overall_ginico:.4f}')
        else:
            print(f'Current bins status: {bins_down}')
            print(f'Overall ginico is: {overall_ginico}')
        # ==================================== Show the whole house ==========================================
        dir_pth = f'data/{date}/{time_stamp}_data/results'
        compose_whole_image(whole_img_dict, dir_pth, roi_stitching_region, bins_down,
                            bins_down_confidence_score, overall_ginico, time_stamp,
                            show_img=False)
        info = {'time': time_stamp, 'bins down': bins_down,
                         'bins down Confidence score': bins_down_confidence_score,
                         'Concentration score': overall_ginico}
        for cam_num in sorted(masks_per_camNum.keys()):
            try:
                info[cam_num] = ginicos_dict_per_cam[cam_num]
            except KeyError:
                info[cam_num] = -100
        # ===================================== final_output ======================================
        analysis.append(info)
    #     ========================================================================================
    analysis_info = ['time', 'bins down', 'bins down Confidence score', 'Concentration score']
    for cam_num in sorted(masks_per_camNum.keys()):
        analysis_info.append(cam_num)
    # with open('results/Analysis.csv', 'w') as csvfile:
    with open(f'results/Analysis_{date}.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=analysis_info)
        writer.writeheader()
        writer.writerows(analysis)
if __name__=='__main__':
    size_limit_Lg_num = 8
    size_limit_sm_num = 700

    den_in_rois = 0
    counts = 0
    bins_up = None
    # ============================== roi call-up ==========================================================
    with open(f"parameters/roi_chkn_cnt_region.json", "r") as file:
        roi_chkn_cnt_region = json.load(file)  # key: image cam-num, value: cv2 image values
    with open(f"parameters/roi_stitching_region.json", "r") as file:
        roi_stitching_region = json.load(file)  # key: image cam-num, value: cv2 image values
    with open(f"parameters/roi_bins_down_region.json", "r") as file:
        roi_bins_down_region = json.load(file)  # key: image cam-num, value: cv2 image values
    with open(f"parameters/roi_foodbin_slate_0_1.json", "r") as file:
        roi_foodbin_slate_0_1 = json.load(file)  # key: image cam-num, value: cv2 image values
    with open(f"parameters/roi_foodbin_ground_2.json", "r") as file:
        roi_foodbin_ground_2 = json.load(file)  # key: image cam-num, value: cv2 image values
    with open(f"parameters/roi_foodbin_ground_3.json", "r") as file:
        roi_foodbin_ground_3 = json.load(file)  # key: image cam-num, value: cv2 image values
    # =================================================================================================
    yellow_pts_thres = 50
    voting_ratio_thr = 0.3
    text = None
    iqr, ginico = None, None
    preproc = 1
    main()
