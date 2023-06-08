import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# fn = 'data/140006_2-11_etc.jpg'
# fn = 'data/161525_2-11_feeding.jpg'
# fn = 'data/110008_2-11_etc.jpg'

fns = glob.glob(os.path.join('data/100005_data', '*.jpg'))
result_dir = 'results/100005_food_bins'
for fn in fns:
    def extract_food_bins(fn):
        image = cv2.imread(fn)
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([34, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return mask_yellow
    # fn = 'data/161525_2-11_feeding.jpg'
    image = cv2.imread(fn)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper threshold for yellow-ish pixels (adjust as needed)
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([40, 255, 255])

    # Alternative range 1
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([35, 255, 255])

    # Alternative range 2
    # lower_yellow = np.array([20, 80, 80])
    # upper_yellow = np.array([40, 255, 255])

    # Alternative range 3 == the best by far
    # lower_yellow = np.array([15, 100, 100])
    # upper_yellow = np.array([35, 255, 255])

    # Alternative range 4
    # lower_yellow = np.array([15, 100, 100])
    # upper_yellow = np.array([34, 255, 255])

    # Create a binary mask for yellow-ish pixels
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # mask_yellow = cv2.resize(mask_yellow, (image.shape[1], image.shape[0]))
    # Create a heatmap by visualizing the mask
    # cv2.imshow("Image", mask_yellow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.imshow(mask_yellow, cmap='hot')
    # # plt.imshow(mask_yellow)
    # plt.axis('off')  # Disable axis
    # plt.show()
    # plt.colorbar().remove()
    # # Save the figure to a temporary image file
    # tmp_file = 'heatmap_temp.png'
    # plt.savefig(tmp_file, bbox_inches='tight', pad_inches=0)  # Save without extra padding
    #
    # # Read the saved image using OpenCV
    # image = cv2.imread(tmp_file)
    #
    # # Crop the image to remove the colorbar area
    # height, width, _ = image.shape
    # cropped_image = image[:, :width - 80]  # Adjust the cropping width as needed
    #
    # # Save the cropped image without the colorbar
    # cv2.imwrite('results/heatmap.png', cropped_image)
    #
    # # Remove the temporary image file
    # import os
    # os.remove(tmp_file)
    # Add colorbar
    # plt.colorbar(label='Yellow-ish Pixel Count')
    save_pth = os.path.join(result_dir, '_'.join(fn.split('/')[-1].split('_')[:2])) + '.jpg'
    cv2.imwrite(save_pth, mask_yellow)
    print(save_pth)
    continue
    # plt.savefig(save_pth)
    # Show the plot
    # plt.show()



