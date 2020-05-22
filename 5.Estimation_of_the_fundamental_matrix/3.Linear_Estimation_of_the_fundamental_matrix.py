"""
Linear Estimation of the fundamental matrix - DLT
@author: Szu-Hao Wu
"""
import numpy as np
import time

def DisplayResults(F, title):
    print(title+' =')
    print(F/np.linalg.norm(F)*np.sign(F[-1,-1]))

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

def DLT(x1, x2, normalize=True):
    # Inputs:
    #    x1 - inhomogeneous inlier correspondences in image 1
    #    x2 - inhomogeneous inlier correspondences in image 2
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    F - the DLT estimate of the fundamental matrix  
    
    # data normalization
    if normalize:
        print('normalize')
        x1, T1 = Normalize(x1)
        x2, T2 = Normalize(x2)
        
    A = np.zeros((x1.shape[1], 9))
    for i in range(x1.shape[1]):
        A[i:i+1,:] = np.kron(x2[:,i:i+1].T, x1[:,i:i+1].T)
    
    U,S,V = np.linalg.svd(A)
    f = V[8:9,:]
    F = np.reshape(f, (3,3))
    
    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    S = np.diag(S)
    F = U@S@V
    
    # data denormalization
    if normalize:
        print('denormalize')
        F = T2.T@F@T1
    
    #F = F/np.linalg.norm(F)
    return F

if __name__ == '__main__':
    # compute the linear estimate with data normalization
    print ('DLT with Data Normalization')
    time_start=time.time()
    F_DLT = DLT(xin1, xin2, normalize=True) #inlier 2D points from outlier rejection
    time_total=time.time()-time_start
    
    # display the resulting F_DLT, scaled with its frobenius norm
    DisplayResults(F_DLT, 'F_DLT')