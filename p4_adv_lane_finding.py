from util_camera import qCamera
from util_vision import qVision




def main():


    output_dir = 'udacity/'

    camera = qCamera()

    num_of_calibd = camera.calibrateSamples('udacity/camera_cal/')
    print('Number of calibrated images: ', num_of_calibd)



    driver_vision = qVision()




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
    