import numpy as np


def calc2ndOrderPoly(fit_coef, yvals):

    assert(fit_coef.size == 3)
    fitx = fit_coef[0]*yvals**2 + fit_coef[1]*yvals + fit_coef[2]

    return fitx



#to keep track of recent detections and to perform sanity checks.
class qLine:
    def __init__(self, np_x=None, np_y=None,  np_fit_coef_x=None, np_fitx = None):
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
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(self.np_y*ym_per_pix, self.np_x*xm_per_pix, 2)

        y_eval = np.max(self.np_y) #the maximum y-value, corresponding to the bottom of the image

        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])

        return curverad