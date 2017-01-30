import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from datetime import timedelta, datetime
import pytz

import os
import cv2

from util_line import qLine

logger.info('util_debug.py loaded')

def DBG_getTimeString():
    dt_now = datetime.now(pytz.timezone('US/Eastern'))
    # dt_now = datetime.now(pytz.timezone('US/Pacific'))

    str_time = dt_now.strftime("%Y%m%d_%H%M%S")

    return str_time

def DBG_saveTimeStampedImg(img, img_name, relative_path):


    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    str_time = DBG_getTimeString()


    file_name =  str_time + '_'+ img_name + '.jpg'

    file_loc = os.path.join(relative_path, file_name) 

    cv2.imwrite(file_loc, img)

    return file_loc

def main():
    print('DBG_getTimeString() returns: ', DBG_getTimeString())

if __name__ == "__main__": 
    import time
    from datetime import timedelta
    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    

def DBG_visualizeDetectedLane(np_x_val_left, np_y_val_left, np_x_val_right, np_y_val_right, left_fitx, right_fitx, file_to_save = ''):
    leftx = np_x_val_left
    rightx = np_x_val_right



    # Plot up the fake data
    plt.plot(leftx, np_y_val_left, 'o', color='red')
    plt.plot(rightx, np_y_val_right, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, np_y_val_left, color='green', linewidth=3)
    plt.plot(right_fitx, np_y_val_right, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.title('DBG_visualizeDetectedLane() - polyfit result')

    if '' == file_to_save:
        plt.show()   
    else:
        import os 
        path, filename = os.path.split(file_to_save)    
        plt.title(filename)
        plt.savefig(file_to_save)

    plt.clf()