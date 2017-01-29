import logging

from datetime import timedelta, datetime
import pytz

import os
import cv2

import logging
logger = logging.getLogger(__name__)

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
    