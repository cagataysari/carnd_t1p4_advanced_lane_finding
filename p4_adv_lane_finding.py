import logging
logging.basicConfig(filename='log_lanefinding.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from util_camera import qCamera
from util_vision import qVision
import pickle
import os.path
import cv2
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

camera = qCamera()
vision = qVision()


num_of_calbd_img = camera.calibrateSamples('udacity/camera_cal/')
logger.info('Camera calibrated')
logger.info('Number of calibrated img: '+ str(num_of_calbd_img) )

def DBG_process_image(img):


        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_bgr = camera.undistortImg(img_bgr)
        img_bgr = vision.highlightLane(img_bgr)

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img


class qLaneFinder:
    def __init__(self):

        self.vision = qVision()
        self.camera = qCamera()

        #load camera data
        if os.path.isfile('calibrated_camera.pickle') :
            with open('calibrated_camera.pickle', 'rb') as handle:
                camera = pickle.load(handle)
        else: 
            camera = qCamera()
            camera.calibrateSamples('udacity/camera_cal/')

            with open('calibrated_camera.pickle', 'wb') as handle:
                pickle.dump(camera, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def processVideo(self, filename_in, filename_out):
        clip1 = VideoFileClip(filename_in)
        output_clip = clip1.fl_image(DBG_process_image) #NOTE: this function expects color images!!
        output_clip.write_videofile(filename_out, audio=False)

    def process_image(self, img):
        """process video frame image
        
        Args:
            img (TYPE):  a w*h*3 RGB array. (720, 1280, 3)
        
        Returns:
            TYPE: Description

        """
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_bgr = self.camera.undistortImg(img_bgr)
        img_bgr = self.vision.highlightLane(img_bgr)

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img

def main():

    lane_finder = qLaneFinder()


    # num_of_calibd = camera.calibrateSamples('udacity/camera_cal/')
    # print('Number of calibrated images: ', num_of_calibd)





    lane_finder.processVideo('udacity/project_video.mp4', 'udacity/Processsed_project_video.mp4')

    # output_name = 'white.mp4'
    # clip1 = VideoFileClip("solidWhiteRight.mp4")
    # white_clip = clip1.fl_image(driver_vision.processImg) #NOTE: this function expects color images!!
    # white_clip.write_videofile(output_name, audio=False)












if __name__ == "__main__": 
    import time
    from datetime import timedelta
    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    