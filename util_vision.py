import numpy as np
import cv2

# a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    
    if 'x' == orient:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    
    sobel_abs = np.absolute(sobel) # the edge detection only cares about absolute value
    num_max_sobel = np.max(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_norm = (255*sobel_abs)/num_max_sobel
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sobel_mask = np.zeros_like(sobel_norm)
    sobel_mask[(sobel_norm>=thresh_min) & (sobel_norm<=thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sobel_mask
    return binary_output


# a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 1) Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    sobel_xy = np.sqrt( np.square(sobel_x) + np.square(sobel_y))
    # 3) Take the absolute value of the derivative or gradient
    sobel = sobel_xy
    sobel_abs = np.absolute(sobel) # the edge detection only cares about absolute value
    num_max_sobel = np.max(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_norm = ((255*sobel_abs)/num_max_sobel).astype(np.uint8) 
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sobel_mask = np.zeros_like(sobel_norm)
    
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    sobel_mask[(sobel_norm>=thresh_min) & (sobel_norm<=thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    binary_output = sobel_mask
    
    return binary_output

# a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)    
    # 3) Take the absolute value of the x and y gradients
    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_angle = np.arctan2(sobel_y_abs , sobel_x_abs)
    # 5) Create a binary mask where direction thresholds are met
    sobel_masked = np.zeros_like(sobel_angle)
    
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sobel_masked[(sobel_angle>=thresh_min) & (sobel_angle<=thresh_max)] = 1
        
    # 6) Return this mask as your binary_output image
    binary_output = sobel_masked
    return binary_output


# a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    ch_satu = img_hls[:,:,2]
    img_1ch = ch_satu
    binary_ch = np.zeros_like(img_1ch)
    thre_min = thresh[0]
    thre_max = thresh[1]
    binary_ch[ (img_1ch>thre_min) & (img_1ch<=thre_max)] =1
    # 3) Return a binary image of threshold result
    binary_output = binary_ch
    

    return binary_output


class qVision:
    def __init__(self):
        pass

    def processImg(self, img_rgb):

        image = np.copy(img_rgb)
        # image =  img_rgb

        hls_binary = hls_select(img_rgb, thresh=(100, 255))
        image[(hls_binary != 1)]  = 0
        # Run the function
        mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(33, 150))
        dir_binary = dir_threshold(image, sobel_kernel=9, thresh=(np.pi*20.0/180.0, np.pi*80.0/180.0)) 
        # dir_binary = dir_threshold(mag_binary, sobel_kernel=11, thresh=(0.7, 1.3), enable_binary_map=True) 


        combined = np.zeros_like(dir_binary)
        # combined[(hls_binary == 1) ] = 1
        combined[  (dir_binary == 1) & (mag_binary == 1)   ] = 1
        # combined[ (mag_binary == 1)   ] = 1
        # Plot the result


        img_final = combined
        return img_final




def DBG_CompareImages(img1, img2, title1, title2, cmap2=None, save_to_file=''):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(img2, cmap2)
    ax2.set_title(title2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if '' != save_to_file: 
        plt.savefig(output_dir+save_to_file)
    plt.show()


def main():

    vision = qVision()


    img_bgr = cv2.imread('udacity/test_images/test4.jpg')

    print('img shape: ', img_bgr.shape)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_procd = vision.processImg(img_rgb)

    DBG_CompareImages(img_rgb, img_procd, 'Original Image', 'Processed Image', cmap2='gray')

# test image: /udacity/test_images/test4.jpg


if __name__ == "__main__": 
    import time
    from datetime import timedelta
    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
