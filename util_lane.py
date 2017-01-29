import numpy as np
from util_line import qLine, calc2ndOrderPoly


class qLane:
    def __init__(self, bottom_pixel_pos=None ):

        self.bottom_pixel_pos = bottom_pixel_pos

        self.left_line = qLine()
        self.right_line = qLine()


        self.y_pixels = np.array([])



    def update(self, left_line, right_line, bottom_pixel_pos):

        if self.isLaneValid(left_line, right_line):

            self.bottom_pixel_pos = bottom_pixel_pos



            left_top_y = min(left_line.getPixelsY())
            right_top_y = min(right_line.getPixelsY())


            top_y = min(left_top_y, right_top_y) 
            bottom_y = self.bottom_pixel_pos 
            self.y_pixels = np.arange(top_y, bottom_y )

            self.updateLeftLine(left_line)
            self.updateRightLine(right_line)

    def updateLeftLine(self, line):

        fit = line.getFitXCoef()
        yvals = self.y_pixels        

        fitx = calc2ndOrderPoly(fit, yvals)

        self.left_line.update(fitx, yvals, fit)


    def updateRightLine(self, line):

        fit = line.getFitXCoef()
        yvals = self.y_pixels        

        fitx = calc2ndOrderPoly(fit, yvals)

        self.right_line.update(fitx, yvals, fit)


    def getLeftLine(self):
        return self.left_line

    def getRightLine(self):
        return self.right_line

    def isLaneValid(self, left_line, right_line):

        is_valid = True
        
        if left_line.isEmpty() or right_line.isEmpty():
            is_valid = False

        return is_valid 


def main():

    top_y_left = 100
    top_y_right = 300

    pixel_num = 200

    top_y = min(top_y_left, top_y_right)
    bottom_pixel_pos = 1000

    left_line = qLine()
    x = np.arange(0,0+pixel_num)
    y = np.arange( top_y_left,top_y_left+pixel_num,1)
    coef = np.array([1,2,3])
    left_line.update(x,y,coef)


    right_line = qLine()
    x = np.arange(500,500+pixel_num)
    y = np.arange(top_y_right,top_y_right+pixel_num, 1)
    coef = np.array([1,2,3])
    right_line.update(x,y,coef)



    lane = qLane(700)
    lane1 = qLane()

    lane.update( left_line, right_line, bottom_pixel_pos)

    projected_left_line = lane.getLeftLine()
    projected_right_line = lane.getRightLine()

    print('projected_left_line x shape: ', projected_left_line.getPixelsX().shape)
    print('projected_right_line x shape: ', projected_right_line.getPixelsX().shape)

    assert(projected_left_line.getPixelsX().size == (bottom_pixel_pos - top_y) ), \
             "\nprojected_left_line.getPixelsX().size: %d \n (bottom_pixel_pos - top_y) :  %d" % (projected_left_line.getPixelsX().size , (bottom_pixel_pos - top_y) )
    assert(projected_left_line.getPixelsY().size == (bottom_pixel_pos - top_y) ), \
             "\nprojected_left_line.getPixelsY().size: %d \n (bottom_pixel_pos - top_y) :  %d" % (projected_left_line.getPixelsY().size , (bottom_pixel_pos - top_y) )



    assert(projected_right_line.getPixelsX().size == (bottom_pixel_pos - top_y) ) , \
            "\n projected_right_line.getPixelsX().size: %d \n (bottom_pixel_pos - top_y) :  %d" % (projected_right_line.getPixelsX().size , (bottom_pixel_pos - top_y) )

    assert(projected_right_line.getPixelsY().size == (bottom_pixel_pos - top_y) ) , \
            "\n projected_right_line.getPixelsY().size: %d \n (bottom_pixel_pos - top_y) :  %d" % (projected_right_line.getPixelsY().size , (bottom_pixel_pos - top_y) )

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    