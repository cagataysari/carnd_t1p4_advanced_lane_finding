import numpy as np


def calc2ndOrderPoly(fit_coef, yvals):

    assert(fit_coef.size == 3)
    fitx = fit_coef[0]*yvals**2 + fit_coef[1]*yvals + fit_coef[2]

    return fitx



#to keep track of recent detections and to perform sanity checks.
class qLine:

    # Define conversions in x and y from pixels space to meters
    YM_PER_PIX = 30/720 # meters per pixel in y dimension
    XM_PER_PIX = 3.7/700 # meteres per pixel in x dimension

    def __init__(self, np_x=np.array([]), np_y=np.array([]),  np_fit_coef_x=np.array([]), np_fitx = np.array([])):
        self.np_x = np_x
        self.np_y = np_y
        self.np_fitx = np_fitx
        self.np_fit_coef_x = np_fit_coef_x

        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.avgx = None     
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None



    def getFittedX(self):
        return self.np_fitx

    def getFitXCoef(self):
        return self.np_fit_coef_x

    def getPixelsX(self):
        return self.np_x
        
    def getPixelsY(self):
        return self.np_y

    def isEmpty(self):
        if self.np_x.size == 0 or self.np_y.size == 0:
            return True
        else:
            return False

    # Checking that they have similar curvature
    # Checking that they are separated by approximately the right distance horizontally
    # Checking that they are roughly parallel
    def update(self, np_x, np_y,  np_fit_coef_x):

        if self.isLineValie(np_x, np_y,  np_fit_coef_x):
            self.np_x = np_x
            self.np_y = np_y
            self.np_fit_coef_x = np_fit_coef_x
            self.np_fitx = calc2ndOrderPoly(np_fit_coef_x, np_y )

    def updateWithObj(self,qline):

        np_x = qline.getPixelsX()
        np_y = qline.getPixelsY()
        np_fit_coef_x = qline.getFitXCoef()

        self.update( np_x, np_y,  np_fit_coef_x )


    def isLineValie(self, np_x, np_y,  np_fit_coef_x):
        return True

    def getCurvatureRadiusInMeters(self):
        """Calculate curvature
           based on Udacity course material
        
        Returns:
            TYPE: Description
        """


        fit_cr = np.polyfit(self.np_y*qLine.YM_PER_PIX, self.np_x*qLine.XM_PER_PIX, 2)

        y_eval = np.max(self.np_y) #the maximum y-value, corresponding to the bottom of the image

        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])

        return curverad

    def getMetersPerPixelInX(self):
        return qLine.XM_PER_PIX


    def getMetersPerPixelInY(self):
        return qLine.YM_PER_PIX



def main():


    pixel_num = 200
    bottom_pixel_pos = 1000

    x_right_start = 500

    top_y = 100

    left_line = qLine()
    # x = np.arange(0,0+pixel_num)
    y = np.linspace( bottom_pixel_pos, top_y,pixel_num)
    coef = np.array([0.0003,0.003,0.8])
    x = calc2ndOrderPoly(coef, y)

    coef = np.polyfit(y, x, 2)
    left_line.update(x,y,coef)
    

    print('Left Lane Curvature: ', left_line.getCurvatureRadiusInMeters())

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    