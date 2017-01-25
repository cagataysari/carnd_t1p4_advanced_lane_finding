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


def findLinePixels(img, initial_x_center, window_width=60, window_height=20, search_step = 10, num_of_windows=2, debug=True):
    """find the line pixels
    
    Args:
        img (TYPE): Description
        starting_x_pos (TYPE): Description
        window_width (int, optional): Description
        window_height (int, optional): Description
        search_step: moving steps of the search window
        num_of_windows: search space consists a number of side-by-side windows

    Returns:
        tuple of list: a list of x positions, and a list of y positions.
    """
    img_height = img.shape[0] # image height
    img_width = img.shape[1]




    y_start_range = range(img_height, int(img_height/2), -window_height) #scan bottom half; y_top ==0

    for y_start in y_start_range:

        # initialize the searching x_pos position
        assert(img_width> num_of_windows*window_width) # search space are two side-by-side windows
        if (initial_x_center-window_width) < 0: #no space on the left
            x_pos = 0
        elif (initial_x_center+window_width) > img_width: # no space on the right
            x_pos = img_width - num_of_windows*window_width 
        else:
            x_pos = initial_x_center - window_width


        ls_btm_left_pos = []
        ls_window_sum = []

        # moving window scanning left to right to find the line center block
        x_start_range = range(x_pos, x_pos + num_of_windows*window_width, search_step)

        for btm_left_x in x_start_range:

            btm_left_y =  y_start

            btm_left_row_idx = btm_left_y  # y denotes row number
            btm_left_col_idx = btm_left_x

            ls_btm_left_pos.append( (btm_left_row_idx,btm_left_col_idx))

            img_window = img[  (btm_left_row_idx-window_height):btm_left_row_idx , 
                                btm_left_col_idx:(btm_left_col_idx+window_width)]

            ls_window_sum.append(np.sum(img_window)) # sum up the numbers of valid line pixels 
            if debug:
                print('btm_left_x =', btm_left_x)
                print('btm_left_y =', btm_left_y)
                print('img_window : ', np.sum(img_window))

                plt.imshow(img_window,'gray')
                plt.show()

        #evaluate after the search
        idx_search = np.argsort(ls_window_sum) [::-1] #reverse to have decending sort

        btm_left_row_idx, btm_left_col_idx  = ls_btm_left_pos[idx_search[0]]

        img_window = img[  (btm_left_row_idx-window_height):btm_left_row_idx , 
                            btm_left_col_idx:(btm_left_col_idx+window_width)]


        initial_x_center = btm_left_col_idx + int(window_width/2)

        if debug:
            print('line is at '+ str(idx_search[0]+1) + 'th block!!!')
            print('ls_btm_left_pos: ' ,ls_btm_left_pos)
            print('left_right_pos found: ', ls_btm_left_pos[idx_search[0]] )

            plt.imshow(img_window,'gray')
            plt.title(' the line center found')
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
