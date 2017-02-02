import numpy as np
from util_line import qLine, calc2ndOrderPoly


class qLane:
    def __init__(self, bottom_pixel_pos=None ):

        self.bottom_pixel_pos = bottom_pixel_pos

        self.left_line = qLine()
        self.right_line = qLine()


        self.y_pixels = np.array([])

        self.current_left_fidelity = 0
        self.current_right_fidelity = 0

        self.curv_fifo = [0]
        self.depart_fifo = [0]

    def update(self, left_line, right_line, bottom_pixel_pos):
        """project line to image bottom. 
        
        Args:
            left_line (TYPE): Description
            right_line (TYPE): Description
            bottom_pixel_pos (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if self.isLaneValid(left_line, right_line):

            self.bottom_pixel_pos = bottom_pixel_pos



            left_top_y = min(left_line.getPixelsY())
            right_top_y = min(right_line.getPixelsY())


            top_y = min(left_top_y, right_top_y) 
            bottom_y = self.bottom_pixel_pos 
            self.y_pixels = np.arange(top_y, bottom_y )

            self.updateLeftLine(left_line)
            self.updateRightLine(right_line)

            self.current_left_fidelity = left_line.getDataFidelity()
            self.current_right_fidelity = right_line.getDataFidelity()

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

        if abs(left_line.getCurvatureRadiusInMeters() - right_line.getCurvatureRadiusInMeters()) > 2000:
            is_vaild = False

        return is_valid 

    def getCarDepartureFromLaneCeterInMeters(self, car_center_pos_pixel):
        """Calculate the car departure distance from lane center 
        
        Args:
            car_center_pos_pixel (TYPE): the pixel position of car center. It would be the image center, if the camera is mounted at center.
        
        Returns:
            float: distance in meteres
        """
        dist_meters = 0

        #check if there are lines captured
        if (self.left_line.isEmpty() is not True) and (self.right_line.isEmpty() is not True):

            left_x_fit_coef = self.left_line.getFitXCoef()

            left_x = calc2ndOrderPoly(left_x_fit_coef, self.bottom_pixel_pos)

            right_x_fit_coef = self.right_line.getFitXCoef()

            right_x = calc2ndOrderPoly(right_x_fit_coef, self.bottom_pixel_pos)

            lane_center_pos_pixel = (left_x+right_x)/2.0

            dist_pixel = car_center_pos_pixel - lane_center_pos_pixel

            dist_meters = dist_pixel* self.left_line.getMetersPerPixelInX()

            # print('left_x: ', left_x)
            # print('right_x: ', right_x)
            # print('lane_center_pos_pixel: ', lane_center_pos_pixel)
            # print('car_center_pos_pixel: ', car_center_pos_pixel)
            # print('dist_pixel: ', dist_pixel)

        dist_meters = self.lowpass(dist_meters, self.depart_fifo)
        return dist_meters  

    def getCurvatureRadiusInMeters(self):

        curv = 0
        if self.current_left_fidelity > self.current_right_fidelity:
            curv = self.left_line.getCurvatureRadiusInMeters() 
        else:
            curv = self.right_line.getCurvatureRadiusInMeters() 

        curv = self.lowpass(curv, self.curv_fifo)

        return curv

    def lowpass(self, data, fifo):

        low_pass_length = 5

        fifo.append(data)

        while len(fifo) > low_pass_length:
            fifo.pop(0)

        return np.average(fifo)


def main():

    top_y_left = 100
    top_y_right = 300

    pixel_num = 200
    bottom_pixel_pos = 1000

    x_right_start = 500

    top_y = min(top_y_left, top_y_right)

    left_line = qLine()
    x = np.arange(0,0+pixel_num)
    y = np.linspace( bottom_pixel_pos, top_y_left,pixel_num)
    # coef = np.array([0,-1,0.3])
    coef = np.polyfit(y, x, 2)
    left_line.update(x,y,coef)


    right_line = qLine()
    x = np.arange(x_right_start,x_right_start+pixel_num)
    y = np.linspace(top_y_right,bottom_pixel_pos, pixel_num)
    coef = np.polyfit(y, x, 2)
    right_line.update(x,y,coef)



    lane = qLane(bottom_pixel_pos)
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

    print('Left Line Curvature: ', lane.getLeftLine().getCurvatureRadiusInMeters())
    print('Right Line Curvature: ', lane.getRightLine().getCurvatureRadiusInMeters())

    print('Car Departure from center: ' + str(lane.getCarDepartureFromLaneCeterInMeters( 1280) ) ) # image width of 1280


if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    