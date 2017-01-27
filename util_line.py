import numpy as np

class qLine:
    def __init__(self, np_x, np_y,  coef_fitx, np_fitx,):
        self.np_x = np_x
        self.np_y = np_y
        self.np_fitx = np_fitx
        self.coef_fitx = coef_fitx




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