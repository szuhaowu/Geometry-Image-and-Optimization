"""
Linear Estimation of the Camera Projection Matrix (Uncalibrated camera)
@author: Szu-Hao Wu
"""

import numpy as np
import time

def Homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))


def Dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]


def Normalize(pts):
    # data normalization of n dimensional pts
    #
    # Input:
    #    pts - is in inhomogeneous coordinates
    # Outputs:
    #    pts - data normalized points
    #    T - corresponding transformation matrix
    
    T = np.eye(pts.shape[0]+1)
    if(pts.shape[0]==2):
        mu_x = np.mean(pts[0,:])
        mu_y = np.mean(pts[1,:])
        sigma_x = np.var(pts[0,:])
        sigma_y = np.var(pts[1,:])
        sigma = sigma_x + sigma_y
        s = (2/sigma)**(0.5)
        T[0,0] = s
        T[1,1] = s
        T[0,2] = -s*mu_x
        T[1,2] = -s*mu_y
    elif(pts.shape[0]==3):
        mu_x = np.mean(pts[0,:])
        mu_y = np.mean(pts[1,:])
        mu_z = np.mean(pts[2,:])
        sigma_x = np.var(pts[0,:])
        sigma_y = np.var(pts[1,:])
        sigma_z = np.var(pts[2,:])
        sigma = sigma_x + sigma_y + sigma_z
        s = (3/sigma)**(0.5)
        T[0,0] = s
        T[1,1] = s
        T[2,2] = s
        T[0,3] = -s*mu_x
        T[1,3] = -s*mu_y
        T[2,3] = -s*mu_z
        
    pts = T.dot(Homogenize(pts))
    return pts, T

def ComputeCost(P, x, X):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #
    # Output:
    #    cost - Total reprojection error
    #n = x.shape[1]
    #covarx = np.eye(2*n)
    x_temp = P.dot(Homogenize(X))
    cost = ((x - Dehomogenize(x_temp))**2).sum()
    
    return cost

def leftNullSpace(x):
    n = x.shape[0]
    e1 = np.zeros((n,1))
    e1[0] = 1
    x_norm = np.linalg.norm(x)
    v = x + np.sign(x[0])*x_norm*e1
    Hv = np.eye(n)-2*(v.dot(v.T))/(np.dot(v.T,v))
    return Hv[1:3,:]
    
def DLT(x, X, normalize=True):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    normalize - if True, apply data normalization to x and X
    #
    # Output:
    #    P - the (3x4) DLT estimate of the camera projection matrix
    P = np.eye(3,4)+np.random.randn(3,4)/10
        
    # data normalization
    if normalize:
        x, T = Normalize(x)
        X, U = Normalize(X)
    else:
        x = Homogenize(x)
        X = Homogenize(X)
    
    A = np.zeros((2*(x.shape[1]),12))
    for i in range(x.shape[1]):
        x_left = leftNullSpace(x[:,i:i+1])
        A[2*i:2*(i+1),:] = np.kron(x_left, np.transpose(X[:,i:i+1]))
    
    u,s,v = np.linalg.svd(A)
    P = v[11:12,:].reshape(P.shape)
    
    # data denormalize
    if normalize:
        P = np.linalg.inv(T) @ P @ U
        
    return P

def displayResults(P, x, X, title):
    print(title+' =')
    print (P/np.linalg.norm(P)*np.sign(P[-1,-1]))
    
    
if __name__ == '__main__':
    # load the data
    x=np.loadtxt('points2D.txt').T
    X=np.loadtxt('points3D.txt').T

    # compute the linear estimate without data normalization
    print ('Running DLT without data normalization')
    time_start=time.time()
    P_DLT = DLT(x, X, normalize=False)
    cost = ComputeCost(P_DLT, x, X)
    time_total=time.time()-time_start
    # display the results
    print('took %f secs'%time_total)
    print('Cost=%.9f'%cost)
    
    
    # compute the linear estimate with data normalization
    print ('Running DLT with data normalization')
    time_start=time.time()
    P_DLT = DLT(x, X, normalize=True)
    cost = ComputeCost(P_DLT, x, X)
    time_total=time.time()-time_start
    
    # Report your P_DLT value here!
    displayResults(P_DLT, x, X, 'P_DLT')