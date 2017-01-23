import numpy as np
import cv2
import glob


class qCamera:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None



    def calibrateSamples(self, sample_dir):
        """calibrate camera based on the Chess-pattern samples
        
        Args:
            sample_dir : Description
        
        Returns:
            integer: number of calibrated image samples
        """

        corner_x = 9 #num of conners at x-axis
        corner_y = 6
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((corner_y*corner_x,3), np.float32)
        objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(sample_dir+'/*.*')

        img = cv2.imread(images[0]) # take an image example to get the size
        #TODO: make all images the same size
        img_size =  (img.shape[1], img.shape[0])

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)


                # file_loc = './udacity/output_images/' + 'calibration/' + 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(file_loc, img)

                # cv2.imshow('img', img)
                # cv2.waitKey(500)


        # Do camera calibration given object points and image points
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


        return len(imgpoints)


    def undistortImg(self, img):

        # img = cv2.imread(file_loc)

        #TODO: Check if mtx is initialized
        img_undst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return img_undst



def main():

    import matplotlib.pyplot as plt

    output_dir = 'udacity/output_images/'

    camera = qCamera()

    num_calibr = camera.calibrateSamples('udacity/camera_cal/')
    #TODO: find out if the more chass image calibrated, the better calibration it gets?
    print('Number of images calibrated: ' , num_calibr)

    img = cv2.imread('udacity/camera_cal/calibration3.jpg')
    img_distorted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img_undist = camera.undistortImg(img_distorted)



    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_distorted)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(output_dir+'undistortion_calibration_img.jpg')
    plt.show()





    img = cv2.imread('udacity/test_images/straight_lines1.jpg')
    img_distorted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_undist = camera.undistortImg(img_distorted)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_distorted)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(output_dir+'undistortion_road_img.jpg')
    plt.show()

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    