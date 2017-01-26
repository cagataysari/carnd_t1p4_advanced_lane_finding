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

    
    if len(indexes) < 2:
        # lane is not found
        return (0,0)
    else:
        return (indexes[idx_sort[0]], indexes[idx_sort[1]])


def findLinePixels(img, initial_x_center, window_width=100, window_height=20, search_step = 10, num_of_windows=2, debug=False):
    """find the line pixels
    
    Args:
        img (TYPE): gray images
        starting_x_pos (TYPE): Description
        window_width (int, optional): Description
        window_height (int, optional): Description
        search_step: moving steps of the search window
        num_of_windows: search space consists a number of side-by-side windows

    Returns:
        tuple of list: list of (x, y) coorinates of line pixels.
    """
    img_height = img.shape[0] # image height
    img_width = img.shape[1]

    np_x_cooridinates = np.array([])
    np_y_cooridinates = np.array([])


    y_start_range = range(img_height, 0, -window_height) #scan bottom half; y_top ==0

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

            if btm_left_row_idx-window_height < 0:
                window_height = btm_left_row_idx # if the last layer at the top of image does not have the height to fit the window

            img_window = img[  (btm_left_row_idx-window_height):btm_left_row_idx , 
                                btm_left_col_idx:(btm_left_col_idx+window_width)]

            ls_window_sum.append(np.sum(img_window)) # sum up the numbers of valid line pixels 


        #evaluate after the search
        idx_search = np.argsort(ls_window_sum) [::-1] #reverse to have decending sort

        
        btm_left_row_idx, btm_left_col_idx  = ls_btm_left_pos[idx_search[0]]


        if ls_window_sum[idx_search[0]] > 10: # only update the initial search position if we actually found pixels
            initial_x_center = btm_left_col_idx + int(window_width/2) # for the next search along y-axis

        binary_map_line  = np.zeros_like(img)
        binary_map_line[(btm_left_row_idx-window_height):btm_left_row_idx , 
                            btm_left_col_idx:(btm_left_col_idx+window_width)] =  img[  (btm_left_row_idx-window_height):btm_left_row_idx , 
                                                                                    btm_left_col_idx:(btm_left_col_idx+window_width)]
                            

        coor_xy = cv2.findNonZero(binary_map_line)

        if coor_xy != None:
            x_val = coor_xy[:,:,0]
            x_val = x_val.reshape(-1)
            y_val = coor_xy[:,:,1]
            y_val = y_val.reshape(-1)


        np_x_cooridinates = np.hstack((np_x_cooridinates, x_val))
        np_y_cooridinates = np.hstack((np_y_cooridinates, y_val))

        if debug:
            print('Search result on the lane line mark:')
            print('x_val :', x_val)
            print('y_val :', y_val)
            print('line is at '+ str(idx_search[0]+1) + 'th block!!!')
            print('ls_btm_left_pos: ' ,ls_btm_left_pos)
            print('btm_left_pos found: ', ls_btm_left_pos[idx_search[0]] )


            img_xy = np.zeros_like(img)

            for x,y in zip(np_x_cooridinates, np_y_cooridinates):  
                img_xy[y,x ] = 1


            plt.imshow(img_xy,'gray')

            plt.title('The line center found')
            plt.show()



    return (np_x_cooridinates, np_y_cooridinates)

def findLaneInGrayImg(img_gray, debug=False):

    x_left, x_right = findLinePositions(img_gray)



    np_left_x, np_left_y = findLinePixels(img_gray, x_left, debug=False)



    #right line
    np_right_x, np_right_y = findLinePixels(img_gray, x_right)


    if True == debug: 
        img_gray_3ch = np.dstack([img_gray, img_gray, img_gray])

        img_gray_3ch = np.zeros_like(img_gray_3ch)
        for x,y in zip(np_left_x, np_left_y):  
            row = y
            col = x
            img_gray_3ch[row,col] = [255,0,0]

        implot=plt.imshow(img_gray_3ch)

        plt.title('findLaneInGrayImg() - left line pixels')
        plt.show()

        img_gray_3ch = np.dstack([img_gray, img_gray, img_gray])

        for x,y in zip(np_right_x, np_right_y):  
            img_gray_3ch[y,x ] = [255,0,0]

        implot=plt.imshow(img_gray_3ch)
        # plt.scatter(np_left_x.tolist(), np_left_y.tolist(),  c='r', s=40)
        # plt.plot([1,2,3,4], [1,4,9,16], 'ro')
        plt.title('findLaneInGrayImg() - right line pixels')
        plt.show()


    return np_left_x, np_left_y, np_right_x, np_right_y


def computeLaneLines(np_x_val_left, np_y_val_left, np_x_val_right, np_y_val_right):
    """compute lane lines
    
    Args:
        np_x_val_left (TYPE): Description
        np_y_val_left (TYPE): Description
        np_x_val_right (TYPE): Description
        np_y_val_right (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    yvals = np_y_val_left
    left_fit = np.polyfit(yvals, np_x_val_left, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]

    yvals = np_y_val_right
    right_fit = np.polyfit(yvals, np_x_val_right, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    return left_fit, left_fitx, right_fit, right_fitx

def visualizeDetectedLane(np_x_val_left, np_y_val_left, np_x_val_right, np_y_val_right, left_fitx, right_fitx):
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
    plt.title('visualizeDetectedLane() - polyfit result')

    plt.show()   


class qLine:
    def __init__(self, np_x, np_y,  coef_fitx, np_fitx,):
        self.np_x = np_x
        self.np_y = np_y
        self.np_fitx = np_fitx
        self.coef_fitx = coef_fitx



def findLaneLines(img_gray, debug=False):
    np_left_x, np_left_y, np_right_x, np_right_y  = findLaneInGrayImg(img_gray, debug=debug)

    left_fit, left_fitx, right_fit, right_fitx = computeLaneLines(np_left_x, np_left_y, np_right_x, np_right_y)


    left_line = qLine(np_left_x, np_left_y, left_fit, left_fitx)
    right_line = qLine(np_right_x, np_right_y, right_fit, right_fitx)

    return left_line, right_line

def main():
    img_gray = cv2.imread('udacity/output_images/birds_eye_view/transformed_processed_test1.jpg', cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_gray,'gray')
    plt.show()
    print('img_gray shape', img_gray.shape)
    
    left_line, right_line = findLaneLines(img_gray, debug=True)


    visualizeDetectedLane(left_line.np_x, left_line.np_y, right_line.np_x, right_line.np_y, 
                        left_line.np_fitx, right_line.np_fitx )



if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    