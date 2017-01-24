import numpy as np
import cv2
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot



def findLinePositions(img_bgr):
    """find left and right lane edges 
    
    Args:
        img_bgr (numpy array): Description
    
    Returns:
        tuple: x-axis positions of left and right lane edge lines 
    """

    histogram = np.sum(img_bgr[img_bgr.shape[0]/2:,:], axis=0)

    indexes = peakutils.indexes(histogram.astype(int), thres=.01, min_dist=100)  #100 is the estimated lane mark width, straight or curved lane


    # sort the peaks
    idx_sort = np.argsort( histogram[indexes] ) [::-1]


    

    # x= np.linspace(0, len(histogram)-1, len(histogram))
    # pplot(x, histogram, indexes)
    # plt.title('First estimate')
    # plt.show()

    
    if len(indexes) < 2:
        # lane is not found
        return (0,0)
    else:
        return (indexes[idx_sort[0]], indexes[idx_sort[1]])



def findLinePixels(img, initial_x_pos, window_size = 60 ):
    """find the line pixels
    
    Args:
        img (TYPE): Description
        starting_x_pos (TYPE): Description
        window_size (int, optional): Description
    
    Returns:
        tuple of list: a list of x positions, and a list of y positions.
    """
    y_size = img.shape[0]

    x_pos = initial_x_pos
    y_start_range = range(y_size, int(y_size/2), -window_size) #scan bottom half; y_top ==0

    print(y_start_range)
    for y_start in y_start_range:

        btm_left_x =  int(x_pos - window_size/2 )
        btm_left_y =  y_start

        btm_left_row_idx = btm_left_y  # y denotes row number
        btm_left_col_idx = btm_left_x
        print('btm_left_x =', btm_left_x)
        print('btm_left_y =', btm_left_y)

        #TODO: move window gradually from left to right. find the window that has the most value of sum
        img_window = img[  (btm_left_row_idx-window_size):btm_left_row_idx , 
                            btm_left_col_idx:(btm_left_col_idx+window_size)]
        print('img_window : ', np.sum(img_window))

        plt.imshow(img_window,'gray')
        plt.show()



    return ([],[])

def findLane(img_bgr):

    x_left, x_right = findLinePositions(img_bgr)

    print('left line: ', x_left )
    print('right line: ',  x_right  )


    ls_left_x, ls_left_y = findLinePixels(img_bgr, x_left)


    return img_bgr





img_bgr = cv2.imread('udacity/output_images/birds_eye_view/transformed_processed_test1.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(img_bgr,'gray')
plt.show()
print('img_bgr shape', img_bgr.shape)
img_lines = findLane(img_bgr)

# cv2.imshow('Two Lines:', img_bgr)
# cv2.waitKey()
