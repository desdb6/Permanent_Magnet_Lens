# HoughTransform module to extract lines and circles from sets of points
# Author: Des De Borger (Des.deborger@student.uantwerpen.be)
# Last revision: 26/03/2025
# Deadline: 28/03/2025

import numpy as np
import math
import typing
import matplotlib.pyplot as plt
from Shapes import *

class LineHoughTransform:
    """A class implementing the Hough Transform for lines"""
    def __init__(self, points=[]):
        """ Initialisation of a list of points. """
        try:
            self.points = list(points)
            if not all([type(x) == type(Point()) for x in points]): # Typecheck the elements in the list
                raise TypeError
        except:
            raise TypeError('Error, please input a list of points!')   
    @property
    def points(self):
        """Returns the list of points"""
        return self._points
    @points.setter
    def points(self, points):
        """Sets the list of points"""
        try:
            self._points = list(points)
            if not all([type(x) == type(Point()) for x in points]): # Typecheck the elements in the list
                raise TypeError
        except:
            raise TypeError('Error, please input a list of points!')         
    def hough_transform(self, n=5000, bins=[500,2000], rmax=5, plot=True):
        """Performs a Hough transform on a list of points and plots the image matrix."""
        typeCheck([bins], list)
        typeCheck([n, bins[0], bins[1]], int)
        typeCheck([rmax], float)
        typeCheck([plot], bool)

        hough_matrix=np.zeros(bins).T # Generate empty matrix
        for point in self.points:
            thetas, rs=pointLineHoughTransform(point.x, point.y, n) # Generate array of (r, theta) values for each point
            hough_matrix+=binaryHistogramLine(thetas, rs, bins, rmax) # Rasterise the (r, theta) curve as a matrix, then add it to the total image matrix
        hough_matrix=hough_matrix[:,0:-1] # Remove the last column, as it is identical to the first column
        if plot==True:
            self.plot_matrix(hough_matrix, rmax) # Plot the points and the matrix if asked
        return hough_matrix
    
    def find_lines(self, n=5000, bins=[500,2000], rmax=5, treshold=0.4, plot=True, max_number_of_lines=999, plot_xlim=[-5, 5], plot_ylim=[-5, 5]):
        """Finds the most likely lines"""
        typeCheck([bins, plot_xlim, plot_ylim], list)
        typeCheck([n, bins[0], bins[1], max_number_of_lines], int)
        typeCheck([rmax, treshold, plot_xlim[0], plot_ylim[0], plot_xlim[1], plot_ylim[1]], float)
        typeCheck([plot], bool)

        hough_matrix=self.hough_transform(n, bins, rmax, plot=False) # Do the Hough Transformation to get the image matrix
        hough_matrix_locmax=hough_matrix.copy() # Create a second matrix, in which we will remove the local maxima
        rows, cols=hough_matrix_locmax.shape # Get the dimensions of the image matrix
        global_max=hough_matrix_locmax.max() # Find the global maximum
        lines=[] # Initialise list of lines
        number_of_lines=0 # Add a counter for how many lines are found
        all_lines_found=False
        while all_lines_found==False and number_of_lines<max_number_of_lines:
            max=hough_matrix_locmax.max() # Find next local minimum
            if max>global_max*treshold: # Check if the local mimimum is large enough to be a line
                number_of_lines+=1 # Add a found line to tohe counter
                (locmax_r, locmax_theta)=np.where(hough_matrix_locmax==max) # Find the coordinates of the next local maximum
                locmax_r=locmax_r[0] # In the case of multiple maxima, choose one
                locmax_theta=locmax_theta[0]
                for i in range(-6, 7):
                    for j in range(-6, 7):
                        hough_matrix_locmax[(locmax_r + i) % rows, (locmax_theta + j) % cols] = 0 #Remove local maximum and the values around it
                locmax_r=(locmax_r-bins[1]/2)*rmax*2/bins[1] #Convert units from pixels to actual (r, theta) value
                locmax_theta=(locmax_theta)*np.pi/bins[0]
                #Transform from (r, theta) space to (m, q) space
                if math.isclose(locmax_theta%np.pi, 0): #Check is the line is vertical
                    m=float('Inf')
                    q=locmax_r
                    lines.append(Line(Point(q,0), Point(q, 1)))
                else:
                    m=-1/np.tan(locmax_theta)
                    q=locmax_r/np.sin(locmax_theta)
                    lines.append(Line(Point(0,q), Point(1, m+q)))
            else:
                all_lines_found=True
        if plot==True:
            self.plot_matrix(hough_matrix, rmax, lines, plot_xlim, plot_ylim)
        return lines
    
    def plot_matrix(self, hough_matrix, rmax=5, lines=[], xlim=[-5, 5], ylim=[-5, 5]):
        typeCheck([lines, xlim, ylim], list)
        typeCheck([rmax, xlim[0], ylim[0], xlim[1], ylim[1]], float)
        typeCheck(hough_matrix, list)
        typeCheck(lines, Line)

        """Plots figrue showing points and Hough Transform of points"""
        fig, ax=plt.subplots(1, 2)
        matrixplot=ax[1].imshow(hough_matrix, cmap='hot', extent=[0,np.pi,rmax,-rmax])

        for line in lines:
            line.draw(ax[0], -10, 10)

        for point in self.points:
            point.draw(ax[0], size=3)

        ax[1].invert_yaxis()
        fig.colorbar(matrixplot)
        ax[1].set_xlabel("theta")
        ax[1].set_ylabel("r")
        ax[1].set_aspect('auto')
        ax[1].set_title('Hough transform of dataset')

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].set_aspect('auto')
        ax[0].grid()
        ax[0].set_title('Points in dataset and found lines')
        ax[0].set_xlim(xlim[0], xlim[1])
        ax[0].set_ylim(ylim[0], ylim[1])
        plt.show()
    
class CircleHoughTransform:
    """A class implementing the Hough Transform for circles"""
    def __init__(self, points=[]):       
        """ Initialisation of a list of points. """
        try:
            self.points = list(points)
            if not all([type(x) == type(Point()) for x in points]):
                raise TypeError
        except:
            raise TypeError('Error, please input a list of points!')
    @property
    def points(self):
        """Returns the list of points"""
        return self._points
    @points.setter
    def points(self, points):
        """Sets the list of points"""
        try:
            self._points = list(points)
            if not all([type(x) == type(Point()) for x in points]): # Typecheck the elements in the list
                raise TypeError
        except:
            raise TypeError('Error, please input a list of points!')  
          
    def hough_transform(self, radius=1, n=1000, bins=[250, 250], xlim=[-5, 5], ylim=[-5, 5], plot=True):
        """Performs a Hough transform on a list of points and plots the image matrix."""
        typeCheck([xlim, ylim, bins], list)
        typeCheck([radius, xlim[0], ylim[0], xlim[1], ylim[1]], float)
        typeCheck([n, bins[0], bins[1]], int)
        typeCheck([plot], bool)
        hough_matrix=np.zeros(bins).T # Generate empty matrix
        for point in self.points:
            xs, ys=pointCircleHoughTransform(point.x, point.y, n, radius) # Generate list of (x, y) values for each point
            hough_matrix+=binaryHistogramCircle(xs, ys, bins, xlim, ylim)  # Rasterise the (x, y) curve as a matrix, then add it to the total image matrix
        if plot==True:
            self.plot_matrix(hough_matrix, xlim, ylim) # Plot the points and the matrix if asked
        return hough_matrix

    def find_circles(self, radius=1, n=1000, bins=[250, 250], xlim=[-5, 5], ylim=[-5, 5], treshold=0.4, plot=True, max_number_of_circles=999):
        typeCheck([xlim, ylim, bins], list)
        typeCheck([radius, xlim[0], ylim[0], xlim[1], ylim[1], treshold], float)
        typeCheck([n, bins[0], bins[1], max_number_of_circles], int)
        typeCheck([plot], bool)
        """Finds the most likely circles"""
        hough_matrix=self.hough_transform(radius, n, bins, xlim, ylim, plot=False) # Do the Hough Transformation to get the image matrix
        hough_matrix_locmax=hough_matrix.copy() # Make a copy of the matrix, in which we will remove the local maxima
        rows, cols=hough_matrix.shape # Get the dimensions of the image matrix
        global_max=hough_matrix.max() # Find the global maximum
        circles=[] # Initialise list of cirlces
        number_of_circles=0 # Add a counter for the amount of found circles
        all_circles_found=False
        while all_circles_found==False and number_of_circles<max_number_of_circles:
            max=hough_matrix_locmax.max() # Find next local minimum
            if max>global_max*treshold: # Check if the local mimimum is large enough to be a circle
                number_of_circles+=1 # Increase the counter of the found circles
                (locmax_y, locmax_x)=np.where(hough_matrix_locmax==max) #Find the coordinates of the next local maximum
                locmax_x=locmax_x[0] #In the case of multiple maxima, choose one
                locmax_y=locmax_y[0]
                for i in range(-6, 7):
                    for j in range(-6, 7):
                        hough_matrix_locmax[(locmax_y + i) % rows, (locmax_x + j) % cols] = 0 #Remove local maximum and the values around it
                locmax_x=xlim[0]+(locmax_x*(xlim[1]-xlim[0])/(bins[0])) #Convert units from pixels to actual (x, y) value
                locmax_y=ylim[0]+(locmax_y*(ylim[1]-ylim[0])/(bins[1]))
                circles.append(Circle(Point(locmax_x, locmax_y), radius))
            else:
                all_circles_found=True
                self.plot_matrix(hough_matrix, circles, xlim, ylim)
        return circles
    
    def plot_matrix(self, hough_matrix, circles=[], xlim=[-5, 5], ylim=[-5, 5]):
        """Plots figrue showing points and Hough Transform of points"""
        typeCheck([xlim, ylim, circles], list)
        typeCheck(hough_matrix, list)
        typeCheck([xlim[0], ylim[0], xlim[1], ylim[1]], float)
        typeCheck(circles, Circle)       
        fig, ax=plt.subplots(1, 2)
        matrixplot=ax[1].imshow(np.flipud(hough_matrix), cmap='hot', extent=[xlim[0],xlim[1],ylim[0],ylim[1]])

        for circle in circles:
            circle.draw(ax[0])

        for point in self.points:
            point.draw(ax[0], size=3)
        
        fig.colorbar(matrixplot)
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_aspect('equal')
        ax[1].set_title('Hough transform of dataset')

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].set_aspect('equal')
        ax[0].set_title('Points in dataset and found circles')
        ax[0].grid()
        ax[0].set_xlim(-5, 5)
        ax[0].set_ylim(-5, 5)
        plt.show()

def pointLineHoughTransform(x, y, n=50):
    """Transforms point to the Hough space for lines"""
    typeCheck([x, y], float)   
    typeCheck([n], int)   
    thetas=np.linspace(0, np.pi, n) # Generate theta values according to formula
    rs=x*np.cos(thetas)+y*np.sin(thetas) # Generate r values according to formula
    return thetas, rs

def pointCircleHoughTransform(x, y, n=50, radius=1):
    """Transforms point to the Hough space for circles"""
    typeCheck([x, y, radius], float)   
    typeCheck([n], int)   
    theta_list=np.linspace(0, 2*np.pi, n) # Generate theta values from zero to 2\pi
    x_values=x+radius*np.cos(theta_list) # Generate x values according to formula
    y_values=y+radius*np.sin(theta_list) # Generate y values according to formula
    return x_values, y_values

def binaryHistogramLine(thetas, rs, bins, rmax=10):
    """Creates a binary histogram for a given range of (r, theta) values"""
    typeCheck([thetas, rs, bins], list)   
    typeCheck(list(thetas)+list(rs), float)  
    typeCheck(bins, int)
    hist, _, _ =np.histogram2d(thetas, rs, bins, range=[[0, np.pi], [-rmax, rmax]]) # Generate histogram of (r, theta) values
    return (hist>0).T.astype(int) # Return binary histogram

def binaryHistogramCircle(xs, ys, bins, xlim=[-10, 10], ylim=[-10, 10]):
    """Creates a binary histogram for a given range of (x, y) values"""
    typeCheck([xlim, ylim, bins, xs, ys], list)   
    typeCheck(list(xs)+list(ys), float)  
    typeCheck(bins+xlim+ylim, int)
    hist, _, _ =np.histogram2d(xs, ys, bins, range=[xlim, ylim]) # Generate histogram of (x, y) values
    return (hist>0).T.astype(int) # Return binary histogram

def noiseTest(M, N):
    typeCheck([M, N], int)
    """Performs a Hough line extraction for a line consisting of M points, with M noise points added"""
    print('M=' + str(M) + ', N=' + str(N) + ', ratio=' + str(N/M))
    m=np.random.uniform(-5,5) # Randomize m
    q=np.random.uniform(-5,5) # Randomize q
    generated_line=Line(Point(0,q), Point(1,q+m)) # Generate randomised line
    print('The generated line is: ' + str(generated_line))
    print('Computing Hough Transform...')
    line_points=generated_line.generate_points(-5,5,M) # Generate points on randomised line
    noise_points=[Point(i, j) for (i, j) in zip(np.random.uniform(-5, 5, N), np.random.uniform(-25, 25, N))] # Generate noise points
    points_list=line_points+noise_points # Combine lists
    hough_3=LineHoughTransform(points_list) # Initiialise HoughTransform-object
    hough_3_matrix=hough_3.hough_transform(5000,[500,2000], plot=False) # Generate image matrix
    hough_3_line=hough_3.find_lines(5000,[500,2000], plot=False)[0] # Find most detectable line
    print('The found line is: ' + str(hough_3_line))
    hough_3.plot_matrix(hough_3_matrix, lines=[hough_3_line])

def typeCheck(objects, check_type):
    """Typechecks a list of objects"""
    for object in objects:
        try: 
            if check_type in [float, list]:
                object_to_type=check_type(object)
            elif not isinstance(object, check_type):
                raise TypeError               
        except:
            raise TypeError(f'Error, the variable {object} has to be of type {check_type}!')