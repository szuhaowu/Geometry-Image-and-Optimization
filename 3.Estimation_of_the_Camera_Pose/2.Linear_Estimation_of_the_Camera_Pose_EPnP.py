"""
Linear Estimation of the Camera Pose (Calibrated camera) - EPnP
@author: Szu-Hao Wu

K = / 1545.0966799187809  0                   639.5 \
    | 0                   1545.0966799187809  359.5 |
    \ 0                   0                   1     /
"""

import numpy as np
import time
from scipy.stats import chi2
from Estimation_of_the_Camera_Pose_Outlier_rejection import Homogenize, Dehomogenize,MSAC


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

def EPnP(x, X, K):
    # Inputs:
    #    x - 2D inlier points
    #    X - 3D inlier points
    # Output:
    #    P - normalized camera projection matrix
    R = np.eye(3) 
    t = np.array([[1,0,0]]).T 
    
    
    """your code here"""
    mu_X = np.mean(X, axis = 1).reshape(3,1)
    cov_X = np.cov(X)
    U,D,Vh = np.linalg.svd(cov_X) # D=(3,)
    V = Vh.T
    sigma_X = np.sum(D)
    
    c1 = mu_X
    c2 = mu_X+ V[:,0:1]
    c3 = mu_X+ V[:,1:2]
    c4 = mu_X+ V[:,2:3]
    alpha = np.zeros((4,x.shape[1]))
    m = np.zeros((2*x.shape[1],12))
    for i in range(x.shape[1]):
        b = X[:,i:i+1]-c1
        alpha[1:4,i:i+1] = Vh.dot(b)
        alpha[0,i:i+1] = 1-np.sum(alpha[1:4,i:i+1])
        x_hat = Dehomogenize(np.linalg.inv(K).dot(Homogenize(x[:,i:i+1]))) #x_hat = (2,1)
        m[2*i:2*i+1,:] = np.array([alpha[0,i], 0, -alpha[0,i]*x_hat[0,0], alpha[1,i], 0, -alpha[1,i]*x_hat[0,0], alpha[2,i], 0, -alpha[2,i]*x_hat[0,0], alpha[3,i], 0, -alpha[3,i]*x_hat[0,0]])
        m[2*i+1:2*i+2,:] = np.array([0, alpha[0,i], -alpha[0,i]*x_hat[1,0], 0, alpha[1,i], -alpha[1,i]*x_hat[1,0], 0, alpha[2,i], -alpha[2,i]*x_hat[1,0], 0, alpha[3,i], -alpha[3,i]*x_hat[1,0]])
        
    U,D,Vh = np.linalg.svd(m) 
    V = np.transpose(Vh[11:12,:])
    c_cam1 = V[0:3,:]
    c_cam2 = V[3:6,:]
    c_cam3 = V[6:9,:]
    c_cam4 = V[9:12,:]
    
    X_cam = np.zeros(np.shape(X))
    for i in range(np.shape(x)[1]):
        X_cam[:,i:i+1] = alpha[0,i]*c_cam1 + alpha[1,i]*c_cam2 + alpha[2,i]*c_cam3 + alpha[3,i]*c_cam4
    mu_X_cam = np.mean(X_cam, axis = 1).reshape(3,1)
    cov_X_cam = np.cov(X_cam)
    U,D,Vh = np.linalg.svd(cov_X_cam)
    sigma_X_cam = np.sum(D)
    if mu_X_cam[2] < 0:
        beta = -np.sqrt(sigma_X/sigma_X_cam)
    else:
        beta = np.sqrt(sigma_X/sigma_X_cam)
    X_cam = beta*X_cam
    mu_X_cam = np.mean(X_cam, axis = 1).reshape(3,1)

    #ICP
    q_X_cam = X_cam - mu_X_cam
    q_X = X - mu_X
    W = q_X_cam.dot(q_X.T)
    U,D,Vh = np.linalg.svd(W)
    if np.linalg.det(U)*np.linalg.det(Vh.T) < 0:
        d = np.eye(3)
        d[2,2] = -1
        R = np.dot(np.dot(U,d), Vh)
    else:
        R = np.dot(U, Vh)
    t = mu_X_cam - R.dot(mu_X)
    P = np.concatenate((R, t), axis=1)
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
    
    tic=time.time()
    P_linear = EPnP(x, X, K)
    toc=time.time()
    time_total=toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('R_linear = ')
    print(P_linear[:,0:3])
    print('t_linear = ')
    print(P_linear[:,-1])
    cost = ComputeCost(P_linear, x, X, K)
    print('Cost:',cost)

