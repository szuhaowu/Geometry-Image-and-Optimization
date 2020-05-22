"""
Nonlinear Estimation of the Camera Pose (Calibrated camera) - LM
@author: Szu-Hao Wu

K = / 1545.0966799187809  0                   639.5 \
    | 0                   1545.0966799187809  359.5 |
    \ 0                   0                   1     /
"""

import numpy as np
import time
from scipy.stats import chi2
from Estimation_of_the_Camera_Pose_Outlier_rejection import Homogenize, Dehomogenize,MSAC
from Linear_Estimation_of_the_Camera_Pose_EPnP import EPnP

def Sinc(x):
    if(x==0):
        y = 1
    else:
        y = np.sin(x)/x
    return y

def dSinc(x):
    if(x==0):
        return 0
    else:
        return np.cos(x)/(x)-np.sin(x)/((x)**2)

def s(theta):
    return (1-np.cos(theta))/theta**2

def ds(theta):
    return (theta*np.sin(theta)-2*(1-np.cos(theta)))/theta**3

def dX_rotated(X,w):
    theta = np.linalg.norm(w)
    if theta < 1e-5:
        return skew(-X)
    else:
        return Sinc(theta)*skew(-X) + \
                np.cross(w,X,axis=0).reshape(3,1).dot(w.T)*dSinc(theta)/theta+ \
                np.cross(w,np.cross(w,X,axis=0).reshape(3,1),axis=0).reshape(3,1).dot(w.T)*ds(theta)/theta + \
                s(theta)*(skew(w).dot(skew(-X)) + skew(-np.cross(w,X,axis=0).reshape(3,1)))

    
def skew(w):
    # Returns the skew-symmetrix represenation of a vector
    w_skew = np.zeros((w.shape[0],w.shape[0]))
    w_skew[0,1] = -w[2,0]
    w_skew[0,2] = w[1,0]
    w_skew[1,0] = w[2,0]
    w_skew[1,2] = -w[0,0]
    w_skew[2,0] = -w[1,0]
    w_skew[2,1] = w[0,0]
    
    return w_skew

def Parameterize(R):
    # Parameterizes rotation matrix into its axis-angle representation
    w = np.zeros((3,1))
    wx = logm(R)
    w[0,0] = -wx[1,2].copy()
    w[1,0] = wx[0,2].copy()
    w[2,0] = wx[1,0].copy()
    
    theta = np.linalg.norm(w)
    
    return w, theta


def Deparameterize(w):
    # Deparameterizes to get rotation matrix
    """your code here"""
    theta = np.linalg.norm(w)
    R = np.cos(theta)*np.eye(3) + Sinc(theta)*skew(w) + (1-np.cos(theta))/(theta**2)*w.dot(w.T)
    
    return R

def Costfromepsilon(epsilon,sigmax):
    #Compute cost of LM
    return np.dot(np.dot(np.transpose(epsilon), np.linalg.inv(sigmax)), epsilon)

def ComputeCost(P, x, X, K):
    # Inputs:
    #    P - camera projection matrix
    #    x - 2D groundtruth image points
    #    X - 3D groundtruth scene points
    #    K - camera calibration matrix
    #
    # Output:
    #    cost - total projection error
    n = x.shape[1]
    covarx = np.eye(2*n) # covariance propagation
    x_hat = K@P@Homogenize(X)
    epsilon = x - Dehomogenize(x_hat)
    epsilon = np.reshape(epsilon, (2*n, 1), 'F')
    cost = epsilon.T.dot(np.linalg.inv(covarx)).dot(epsilon)
    
    return float(cost)


def Jacobian(R, w, t, X):
    # compute the jacobian matrix
    # Inputs:
    #    R - 3x3 rotation matrix
    #    w - 3x1 axis-angle parameterization of R
    #    t - 3x1 translation vector
    #    X - 3D inlier points
    #
    # Output:
    #    J - Jacobian matrix of size 2*nx6
    
    x_hat = Dehomogenize(R.dot(X) + t)
    theta = np.linalg.norm(w)
    
    if theta < 1e-5:
        X_rotated = X + np.cross(w,X,axis=0)
    else:
        X_rotated = X + Sinc(theta)*np.cross(w,X,axis=0) + s(theta)*np.cross(w,np.cross(w,X,axis=0),axis=0)
    w_hat = X_rotated[2,:] + t[2,0]
    J = np.zeros((2*X.shape[1],6))
    for i in range(X.shape[1]):
        
        J[2*i:2*i+2,0:3] = np.array([[1/w_hat[i], 0, -x_hat[0,i]/w_hat[i]] , [0, 1/w_hat[i], -x_hat[1,i]/w_hat[i]]]).dot(dX_rotated(X[:,i:i+1],w))
        J[2*i:2*i+2,3:6] = np.array([[1/w_hat[i], 0, -x_hat[0,i]/w_hat[i]] , [0, 1/w_hat[i], -x_hat[1,i]/w_hat[i]]])
        
    return J

def LM(P, x, X, K, max_iters, lam):
    # Inputs:
    #    P - initial estimate of camera pose
    #    x - 2D inliers
    #    X - 3D inliers
    #    K - camera calibration matrix 
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    #
    # Output:
    #    P - Final camera pose obtained after convergence
    n = x.shape[1]
    x_hat = K@P@Homogenize(X)
    epsilon = x - Dehomogenize(x_hat)
    epsilon = np.reshape(epsilon, (2*n, 1), 'F')
    S = K[0:2,0:2]
    sigmax = np.eye(2*n)
    for i in range(n):
        sigmax[2*i:2*i+2,2*i:2*i+2] = S.dot(S.T)
        
    for i in range(max_iters): 
        R = P[:,0:3]
        t = P[:,-1].reshape(3,1)
        w,theta = Parameterize(R)
        J = Jacobian(R,w,t,X) #J = 2n*6
        
        A = (J.T).dot(np.linalg.inv(sigmax)).dot(J) + lam*np.eye(6) #6*6
        b = (J.T).dot(np.linalg.inv(sigmax)).dot(epsilon) #6*1
        delta = np.linalg.inv(A).dot(b)
        
        w0 = w + delta[0:3,0:1]
        t0 = t + delta[3:6,0:1]
        
        R0 = Deparameterize(w0)
        P0 = np.concatenate((R0, t0), axis=1)
        
        x0_hat = R0.dot(X) + t
        epsilon0 = x - Dehomogenize(x0_hat)
        epsilon0 = epsilon0.reshape(2*x.shape[1],-1,order = 'F')
    
        cost_prev = ComputeCost(P, x, X, K)
        cost0 = ComputeCost(P0, x, X, K)
        if(cost0 > cost_prev):
            lam = 10*lam
        
        else:
            cost = cost0
            P = P0
            epsilon = epsilon0
            lam = 0.1*lam

        cost = ComputeCost(P, x, X, K)
        print('iter %03d Cost %.9f'%(i+1, cost))
    
    return P


if __name__ == '__main__':
    tol = 5.5
    p = 0.99
    alpha = 0.95
    thresh = np.sqrt(chi2.ppf(0.95,2))
    K = np.array([[1545.0966799187809, 0, 639.5], 
          [0, 1545.0966799187809, 359.5], 
          [0, 0, 1]])
    
    
    tic=time.time()
    
    x0=np.loadtxt('points2D.txt').T
    X0=np.loadtxt('points3D.txt').T
    
    
    cost_MSAC, P_MSAC, inliers, trials = MSAC(x0, X0, K, thresh, tol, p)
    
    x = x0[:,inliers]
    X = X0[:,inliers]
    
    P_linear = EPnP(x, X, K)
    
    
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    tic = time.time()
    P_LM = LM(P_linear, x, X, K, max_iters, lam)
    w_LM,_ = Parameterize(P_LM[:,0:3])
    toc = time.time()
    time_total = toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('w_LM = ')
    print(w_LM)
    print('R_LM = ')
    print(P_LM[:,0:3])
    print('t_LM = ')
    print(P_LM[:,-1])