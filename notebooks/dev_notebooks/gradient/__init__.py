"""
Color gradient generator
Written by Tomer Burg
Last revised 1/24/2019
Python 3.6
This class creates a gradient object using a range of color intervals matched with
numerical values, and returns a colormap that matches the specified ranges to a
passed array of numerical values.
"""

import numpy as np
import matplotlib.colors as col

#=============================================================================================
# Gradient class
#=============================================================================================

class Gradient:
    
    def __init__(self,*args):
        self.args = args
        
        #Set threshold levels
        self.thres = []
        self.thres_min = []
        
        #Error check arguments
        error = 0
        for arg in args:
            
            #Ensure each argument & subargument has a length of 2
            if len(arg) != 2: error = 1
            if len(arg[0]) != 2: error = 1
            if len(arg[1]) != 2: error = 1
            
            #Ensure the 2nd element of each argument is a number
            if isinstance(arg[0][1], (int, float)) == False: error = 2
            if isinstance(arg[1][1], (int, float)) == False: error = 2
            
            #Ensure that the 1st element of each argument is either a hex str or rgb tuple
            if isinstance(arg[0][0], (str, tuple)) == False: error = 3
            if isinstance(arg[1][0], (str, tuple)) == False: error = 3
            
            #Ensure gradient values are continuous
            if len(self.thres) > 0 and self.thres[-1] != arg[0][1]: error = 4
            
            #Append threshold levels
            self.thres.append(arg[0][1])
            self.thres.append(arg[1][1])
            self.thres_min.append(arg[0][1])
            
        #Ensure values are either constantly increasing or decreasing
        check_thres = np.array(self.thres)
        diff = check_thres[1:] - check_thres[:-1]
        if np.min(diff) == 0 and np.max(diff) > 0:
            pass
        elif np.min(diff) < 0 and np.max(diff) == 0:
            self.thres = self.thres[::-1]
        else:
            error = 4
        
        #Output error messages
        if error == 1: raise RuntimeError('Each argument must have 2 elements, e.g., [["#00FFFF",25.0],["#0000FF",29.0]]')
        if error == 2: raise RuntimeError('The second element must be a number, e.g., [["#00FFFF",25.0]')
        if error == 3: raise RuntimeError('The first element must be a hex string or an rgb tuple, e.g., [["#00FFFF",25.0]')
        if error == 4: raise RuntimeError('Values assigned to the gradient must be continuous, either increasing or decreasing.')
        
    #Returns the hex string corresponding to the passed rgb values
    def rgb(self,r,g,b):
        r = int(r)
        g = int(g)
        b = int(b)
        return '#%02x%02x%02x' % (r, g, b)
        
    #Computes a hex value matching up with the current position relative to the range of colors.
    #position = current position within the range of colors (e.g., 1)
    #rng = range of colors (e.g. 5, so 1/5 would be 20% of the range)
    #col1 = Starting RGB color for the range (e.g. [0,255,255])
    #col2 = Ending RGB color for the range (e.g. [0,0,255])
    def getColor(self,position,rng,col1,col2):
        
        #Retrieve r,g,b values from tuple
        r1,g1,b1 = col1
        r2,g2,b2 = col2
    
        #Get difference in each r,g,b value between start & end 
        rdif = float(r2 - r1)
        gdif = float(g2 - g1)
        bdif = float(b2 - b1)
        
        #Calculate r,g,b values for the specified position within the range
        r3 = r2 + (-1.0 * position * (rdif / float(rng)))
        g3 = g2 + (-1.0 * position * (gdif / float(rng)))
        b3 = b2 + (-1.0 * position * (bdif / float(rng)))
    
        #Return in hex string format
        return self.rgb(r3,g3,b3)

    #Finds the nearest gradient range to use
    def find_nearest(self,arr,val):
        for ival in arr[::-1]:
            if ival <= val:
                return arr.index(ival)
        
    #Create a color map based on passed levels
    def get_cmap(self,levels):
        
        #Add empty color list
        self.colors = []
        
        #Iterate through levels
        for lev in levels:
            
            #Check if level is outside of range
            if lev < self.thres[0]:
                start_hex = self.args[0][0][0]
                if "#" not in start_hex: start_hex = self.rgb(start_hex[0],start_hex[1],start_hex[2])
                self.colors.append(start_hex)
            
            elif lev > self.thres[-1]:
                end_hex = self.args[-1][1][0]
                if "#" not in end_hex: end_hex = self.rgb(end_hex[0],end_hex[1],end_hex[2])
                self.colors.append(end_hex)
                
            else:
                
                #Find closest lower threshold
                idx = self.find_nearest(self.thres_min,lev)
                
                #Retrieve start & end values
                start_value = self.args[idx][0][1]
                end_value = self.args[idx][1][1]
                
                #Calculate start and end RGB tuples, if passed as hex
                start_hex = self.args[idx][1][0]
                end_hex = self.args[idx][0][0]
                if "#" in start_hex:
                    start_hex = start_hex.lstrip('#')
                    end_hex = end_hex.lstrip('#')
                    start_rgb = tuple(int(start_hex[i:i+2], 16) for i in (0, 2 ,4))
                    end_rgb = tuple(int(end_hex[i:i+2], 16) for i in (0, 2 ,4))
                else:
                    start_rgb = start_hex
                    end_rgb = end_hex
    
                #Get hex value for the color at this point in the range
                nrange_color = (end_value - start_value)
                idx = lev - start_value
                hex_val = self.getColor(idx,nrange_color,start_rgb,end_rgb)
                
                #Append color to list
                self.colors.append(hex_val)
        
        #Convert to a colormap and return
        self.cmap = col.ListedColormap(self.colors)
        return self.cmap
