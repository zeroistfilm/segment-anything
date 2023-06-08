import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    if prep==0:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define the threshold for white-ish pixels (adjust as needed)
        threshold = 170

        # Create a binary mask for white-ish pixels
        mask = (gray >= threshold)

        # Create a heatmap by visualizing the mask
        plt.imshow(mask, cmap='hot')

        # Add colorbar
        plt.colorbar(label='White-ish Density')

        # Show the plot
        plt.show()
    elif prep==1:
        from scipy.stats import gaussian_kde

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define the threshold for white-ish pixels (adjust as needed)
        threshold = 170

        # Get the white-ish pixel coordinates
        whiteish_pixels = np.argwhere(gray >= threshold)

        # Extract the x and y coordinates
        x = whiteish_pixels[:, 1]
        y = whiteish_pixels[:, 0]

        # Perform kernel density estimation (KDE) on the white-ish pixel coordinates
        kde = gaussian_kde(np.vstack([x, y]))

        # Create a grid of points over the image
        x_grid, y_grid = np.meshgrid(np.arange(0, gray.shape[1]), np.arange(0, gray.shape[0]))
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])

        # Evaluate the KDE on the grid points
        density_values = kde(grid_points)

        # Reshape the density values to match the image shape
        density_map = density_values.reshape(gray.shape)

        # Create a heatmap by visualizing the density map
        plt.imshow(density_map, cmap='hot')

        # Add colorbar
        plt.colorbar(label='White-ish Density')

        # Show the plot
        plt.show()
    elif prep == 2:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define the threshold for white pixels (adjust as needed)
        threshold = 150

        # Extract white pixels based on the threshold
        white_pixels = np.where(gray >= threshold, 255, 0)

        # Create a mask of white pixels
        white_mask = np.zeros_like(gray)
        white_mask[white_pixels > 0] = 255

        # Show the resulting white pixels
        cv2.imshow("White Pixels", white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif prep == 3:
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper thresholds for white color in HSV
        # lower_white = np.array([0, 0, 200])
        # upper_white = np.array([179, 25, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([250, 25, 255])
        # Create a mask for white pixels
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Apply the mask to extract white pixels
        white_pixels = cv2.bitwise_and(image, image, mask=mask_white)

        # Show the resulting white pixels
        cv2.imshow("White Pixels", white_pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__=='__main__':
    fn = 'data/140006_2-11_etc.jpg'
    # Load the image
    image = cv2.imread(fn)
    prep = 1
    main()
