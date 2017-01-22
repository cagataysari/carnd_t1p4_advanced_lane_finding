from util_camera import qCamera


def main():
    import time
    from datetime import timedelta

    time_start = time.time()

    output_dir = 'udacity/'

    camera = qCamera()

    camera.calibrateSamples('udacity/camera_cal/')






    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    
if __name__ == "__main__": main()