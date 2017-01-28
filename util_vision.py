import numpy as np
from numpy.linalg import inv
import cv2
from util_line import qLine
from util_lane import qLane
from util_findLane import findLaneLines

def extractSatuCh(img):

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    ch_satu = img_hls[:,:,2]

    return ch_satu


# a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    
    # 1) Convert to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = extractSatuCh(image)
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
    
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sobel_mask[(sobel_norm>=thresh_min) & (sobel_norm<=thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    binary_output = sobel_mask
    
    return binary_output


# a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to image
    # 1) Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = extractSatuCh(image)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)    
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
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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

# a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select_s(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    ch_hue = img_hls[:,:,0]
    # ch_satu = img_hls[:,:,2]
    img_1ch = ch_hue
    binary_ch = np.zeros_like(img_1ch)
    thre_min = thresh[0]
    thre_max = thresh[1]
    binary_ch[ (img_1ch>thre_min) & (img_1ch<=thre_max)] =1
    # 3) Return a binary image of threshold result
    binary_output = binary_ch
    

    return binary_output

class qVision:
    def __init__(self):


        # perspective transformation matrix for birds' eye view 
        self.M = None  

        # current lane 
        self.lane = qLane()


    def processImg(self, img_rgb, debug=False):

        # image = np.copy(img_rgb)
        image_orig = np.copy(img_rgb)


        hls_binary = hls_select(image_orig, thresh=(66, 255))
        # image[(hls_binary != 1)]  = 0
        # Run the function
        mag_binary = mag_thresh(image_orig, sobel_kernel=27, thresh=(50, 220 ))

        dir_binary = dir_threshold(image_orig, sobel_kernel=15, thresh=(np.pi*20.0/180.0, np.pi*80.0/180.0)) 
        # dir_binary = dir_threshold(image_orig2, sobel_kernel=15, thresh=(0.7, 1.2) ) # 40~80 degree
        # dir_binary = dir_threshold(mag_binary, sobel_kernel=11, thresh=(0.7, 1.3), enable_binary_map=True) 


        combined = np.zeros_like(dir_binary)
        # combined[(hls_binary == 1) ] = 1
        # combined[  (dir_binary == 1) & (mag_binary == 1)   ] = 1
        combined[ (dir_binary == 1) & (mag_binary == 1) & (hls_binary == 1) ] = 1
        # Plot the result

        if debug == True:
            DBG_CompareThreeGrayImages(hls_binary , mag_binary,  dir_binary ,'HLS satu', 'mag', 'directional' )
        

        img_final = combined
        return img_final



    def transformToBirdsEyeView(self, img):
        """based on Udacity course material
        
        Args:
            img (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        img_size = (img.shape[1], img.shape[0])

        # init M when the first image is fed in
        if None == self.M:
            src = np.float32(
                [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
                [((img_size[0] / 6) - 10), img_size[1]],
                [(img_size[0] * 5 / 6) + 60, img_size[1]],
                [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

            dst = np.float32(
                [[(img_size[0] / 4), 0],
                [(img_size[0] / 4), img_size[1]],
                [(img_size[0] * 3 / 4), img_size[1]],
                [(img_size[0] * 3 / 4), 0]])

            self.M = cv2.getPerspectiveTransform(src, dst)

        #TODO: check if img_size stays the same
        img_warped = cv2.warpPerspective(img, self.M, img_size)

        return img_warped

    def imaginLines(self, img_undist, left_line, right_line):
        """ project lines on the the road
        
        Args:
            img_undist (TYPE): undistorted image to project the lane on 
            left_line (TYPE): Description
            right_line (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        #based on Udacity course material

        # Create an image to draw the lines on
        color_warp = np.zeros_like(img_undist).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_fitx = left_line.getFittedX()
        yvals = left_line.getPixelsY()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])

        right_fitx = right_line.getFittedX()
        yvals = right_line.getPixelsY()
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))


        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = inv(self.M)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img_undist.shape[1], img_undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
        # plt.imshow(result)

        return result

    def hightlightLane(self, img_bgr):
        """hightlight the current lane
        
        Args:
            img_bgr (TYPE): an camera-undistorted road image 
        
        Returns:
            TYPE: Description
        """
        img_bgr_procd = self.processImg(img_bgr)
    
        img_bgr_procd_birdview = self.transformToBirdsEyeView(img_bgr_procd)
        # cv2.imshow('img_bgr_procd_birdview',img_bgr_procd_birdview)
        # cv2.waitKey()
        left_line, right_line = findLaneLines(img_bgr_procd_birdview)

        self.lane.update( left_line, right_line, bottom_pixel_pos=img_bgr.shape[0])

        projected_left_line = self.lane.getLeftLine()
        projected_right_line = self.lane.getRightLine()

        img_imagined_lines = self.imaginLines(img_bgr, projected_left_line, projected_right_line)

        return img_imagined_lines



def DBG_CompareImages(img1, img2, title1, title2, cmap2=None, save_to_file=''):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if 'gray' != cmap2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(img2, cmap2)
    ax2.set_title(title2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if '' != save_to_file: 
        plt.savefig(save_to_file)
    plt.show()
    plt.close()


def DBG_CompareThreeGrayImages(img1, img2, img3, title1, title2, title3 ):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    f.tight_layout()

    ax1.imshow(img1, cmap='gray', vmin = 0, vmax = 1)
    ax1.set_title(title1, fontsize=30)

    ax2.imshow(img2, cmap='gray', vmin = 0, vmax = 1)
    ax2.set_title(title2, fontsize=30)

    ax3.imshow(img3, cmap='gray', vmin = 0, vmax = 1)
    ax3.set_title(title3, fontsize=30)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()


def main():


    #load camera
    from util_camera import qCamera

    #save camera data
    import pickle
    import os.path
    if os.path.isfile('calibrated_camera.pickle') :
        with open('calibrated_camera.pickle', 'rb') as handle:
            camera = pickle.load(handle)
    else: 
        camera = qCamera()
        camera.calibrateSamples('udacity/camera_cal/')

        with open('calibrated_camera.pickle', 'wb') as handle:
            pickle.dump(camera, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ##############################3
    vision = qVision()

    import glob
    import os

    sample_dir = 'udacity/test_images/'
    images_loc = glob.glob(sample_dir+'/*.jpg')

    for img_loc in images_loc:
        img_bgr = cv2.imread(img_loc)

        img_procd = vision.processImg(img_bgr)

        path, filename = os.path.split(img_loc)    
        file_loc = 'udacity/output_images/' + 'test_images/' + 'processed_'+ filename
        cv2.imwrite(file_loc, (img_procd*225).astype(np.uint8) ) #Only 8-bit images can be saved using this function, so convert from (0.0, 1.0) to (0,255)
        # cv2.imshow('Processed Image', img_procd)
        # DBG_CompareImages(img_rgb, img_procd, 'Original Image', 'Processed Image', cmap2='gray')



    img_bgr = cv2.imread('udacity/test_images/test1.jpg' )

    img_procd = vision.processImg(img_bgr)

    DBG_CompareImages(img_bgr, img_procd, 'Original Image', 'Thresholded Binary Image', cmap2='gray')


    ##########################################
    # Test for Birds Eye View Transformation
    ##########################################

        
    img_distorted = cv2.imread('udacity/test_images/straight_lines1.jpg')
    img_undist = camera.undistortImg(img_distorted)
    img_undist_birdview = vision.transformToBirdsEyeView(img_undist)


    DBG_CompareImages(img_distorted, img_undist_birdview, 'Original Image', "Bird's Eye View Image",save_to_file='udacity/output_images/persp_birds_eye_view.jpg')




    ##########################################
    # Test for Birds Eye View Transformation on black and white thresholded images
    ##########################################
    # test on thresholded images
    sample_dir = 'udacity/output_images/test_images/'
    images_loc = glob.glob(sample_dir+'/*.jpg')
    for loc in images_loc:
        bgr = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

        distorted = bgr
        undist = camera.undistortImg(distorted)
        undist_birdview = vision.transformToBirdsEyeView(undist)

        path, filename = os.path.split(loc)    
        file_loc = 'udacity/output_images/' + 'birds_eye_view/' + 'transformed_'+ filename
        cv2.imwrite(file_loc, img_undist_birdview ) #Only 8-bit images can be saved using this function, so convert from (0.0, 1.0) to (0,255)
        # cv2.imshow('Processed Image', img_procd)
        # DBG_CompareImages(img_rgb, img_procd, 'Original Image', 'Processed Image', cmap2='gray')



    ##########################################
    # Test on Lane Finding
    ##########################################

    img_lane_hightlighted = vision.hightlightLane(img_undist)

    DBG_CompareImages(img_undist, img_lane_hightlighted, 'Undistorted Image', "Reimagined Lane")

    sample_dir = 'udacity/test_images/'
    images_loc = glob.glob(sample_dir+'*.jpg')

    for loc in images_loc:
        distorted = cv2.imread(loc)
        undist = camera.undistortImg(distorted)

        lane_hightlighted = vision.hightlightLane(undist)

        path, filename = os.path.split(loc)    
        file_loc = 'udacity/output_images/' + 'hightlighted_lane/' + 'hightlighted'+ filename
        cv2.imwrite(file_loc, lane_hightlighted ) 

if __name__ == "__main__": 
    import time
    from datetime import timedelta
    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
