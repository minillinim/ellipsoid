#!/usr/bin/python

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg
from random import random

class EllipsoidTool:
    """Some stuff for playing with ellipsoids"""
    def __init__(self): pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of 3D points like this:
        P = [[x,y,z],
             [x,y,z],
             [x,y,z]]
        
        Returns:
        (center, radii, rotation)
        
        """
        (N, d) = np.shape(P)
    
        # This method only works well if the points are
        # centered around the origin
        min_P = np.min(P, axis=0)
        P -= min_P
        max_P = np.max(P, axis=0)/2
        P -= max_P
    
        # Q will be out working array
        Q = np.copy(P.T)
        Q = np.vstack([Q, np.ones(N)])
    
        # initializations
        err = 1 + tolerance
        u = np.array([1.0 / N for i in range(N)]) # first iteration
    
        # Khachiyan Algorithm
        while err > tolerance:
            X = np.dot(Q, np.dot(np.diag(u), Q.T))
            M = np.diag(np.dot(Q.T , np.dot(linalg.inv(X), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = np.array([i * (1 - step_size) for i in u])
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u
    
        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = linalg.inv( np.dot(P.T, np.dot(np.diag(u), P)) - np.dot(center, center.T) ) / float(d)
    
        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        # fix P and center
        P += (min_P + max_P)
        center += (min_P + max_P)
        
        return (center, radii, rotation)

    def plotEllipsoid(self, center, radii, rotation, ax=None, plotAxes=False):
        """Plot an ellipsoid"""
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        
        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    
        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
    
    
            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color='r')
    
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
        
        if make_ax:
            plt.show()
            plt.close(fig)
            del fig
        
if __name__ == "__main__":
    # make 100 random points
    P = np.reshape([random()*100 for i in range(300)],(100,3))
    
    # find the ellipsoid
    ET = EllipsoidTool()
    (center, radii, rotation) = ET.getMinVolEllipse(P, .001)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot points
    ax.scatter(P[:,0], P[:,1], P[:,2], color='g', marker='*', s=100)

    # plot ellipsoid
    ET.plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=True)
    
    plt.show()
    plt.close(fig)
    del fig
