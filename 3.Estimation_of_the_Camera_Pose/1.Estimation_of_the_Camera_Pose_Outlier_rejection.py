"""
Estimation of the Camera Pose - Outlier rejection (RANSAC) (Calibrated camera)
@author: Szu-Hao Wu

K = / 1545.0966799187809  0                   639.5 \
    | 0                   1545.0966799187809  359.5 |
    \ 0                   0                   1     /
"""

import numpy as np
import time
from scipy.stats import chi2
import random

def Homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]
    
def ComputeCostInlier(P, x, X, K, tol):
    # Inputs:
    #    P - camera projection matrix
    #    x - 2D groundtruth image points
    #    X - 3D groundtruth scene points
    #    K - camera calibration matrix
    #
    # Output:
    #    cost - total projection error
    n = x.shape[1]
    #covarx = np.eye(2*n) # covariance propagation
    
    error = np.zeros((x.shape[1],))
    num_inliers = 0
    cost = 0
    for i in range(n):
        #print(Homogenize(X[:,i:i+1]).shape)
        x_cam = K@P@Homogenize(X[:,i:i+1])
        x_cam = Dehomogenize(x_cam)
        error[i] = np.linalg.norm(x_cam - x[:,i:i+1])
        if error[i] < tol:
            cost = cost + error[i]
            num_inliers +=1
        else:
            cost = cost + tol
            
    return cost,error,num_inliers

def inlier(P,x,X,K,tol):
    num_inliers = 0
    inliers = []
    for i in range(x.shape[1]):
        x_cam = K@P@Homogenize(X[:,i:i+1])
        x_cam = Dehomogenize(x_cam)
        error = np.linalg.norm(x_cam - x[:,i:i+1])
        if error < tol:
            inliers.append(i)
            num_inliers+=1
            
    return num_inliers,inliers

def sign(x):
    if x < 0:
        return -1;
    elif x > 0:
        return 1;
    else:
        return 0;
    
def model(p1_x,p2_x,p3_x,P1_X,P2_X,P3_X, K, x, X):
    #estimation of the camera pose from random selected 3 points
    #use the 3-point algorithm of Finsterwalder to estimate camera pose
    # Inputs:
    #    p1_x, p2_x, p3_x - 3 of 2D-image points
    #    P1_X, P2_X, P3_X - 3 of 3D-scene points
    #    x - 2D groundtruth image points
    #    X - 3D groundtruth scene points
    #    K - camera calibration matrix
    #
    # Output:
    #    cost - total projection error
    P = np.zeros((3,4))
    f = K[1,1]
    p1_normal = np.linalg.inv(K)@Homogenize(p1_x)
    p2_normal = np.linalg.inv(K)@Homogenize(p2_x)
    p3_normal = np.linalg.inv(K)@Homogenize(p3_x)
    
    a = (np.sqrt((P2_X[0]-P3_X[0])**2 + (P2_X[1]-P3_X[1])**2 + (P2_X[2]-P3_X[2])**2))[0]
    b = (np.sqrt((P1_X[0]-P3_X[0])**2 + (P1_X[1]-P3_X[1])**2 + (P1_X[2]-P3_X[2])**2))[0]
    c = (np.sqrt((P1_X[0]-P2_X[0])**2 + (P1_X[1]-P2_X[1])**2 + (P1_X[2]-P2_X[2])**2))[0]

    u1 = f*p1_normal[0,0]/p1_normal[2,0]
    v1 = f*p1_normal[1,0]/p1_normal[2,0]
    u2 = f*p2_normal[0,0]/p2_normal[2,0]
    v2 = f*p2_normal[1,0]/p2_normal[2,0]
    u3 = f*p3_normal[0,0]/p3_normal[2,0]
    v3 = f*p3_normal[1,0]/p3_normal[2,0]
    
    j1 = (1/np.sqrt(u1**2+v1**2+f**2))*np.array([[u1],[v1],[f]])
    j2 = (1/np.sqrt(u2**2+v2**2+f**2))*np.array([[u2],[v2],[f]])
    j3 = (1/np.sqrt(u3**2+v3**2+f**2))*np.array([[u3],[v3],[f]])
    
    cos_alpha = j2.T.dot(j3)[0,0]
    cos_beta = j1.T.dot(j3)[0,0]
    cos_gamma = j1.T.dot(j2)[0,0]
    sin_alpha = 1 - cos_alpha**2
    sin_beta = 1 - cos_beta**2
    sin_gamma = 1 - cos_gamma**2
    
    G = c**2 * (c**2*sin_beta - b**2*sin_gamma)
    H = b**2 * (b**2-a**2)*sin_gamma + c**2*(c**2 + 2*a**2)*sin_beta + 2*(b**2)*(c**2)*(-1+cos_alpha*cos_beta*cos_gamma)
    I = b**2 * (b**2-c**2)*sin_alpha + a**2*(a**2+2*c**2)*sin_beta + 2*a**2*b**2*(-1+cos_alpha*cos_beta*cos_gamma)
    J = a**2 * (a**2*sin_beta - b**2*sin_alpha)
    
    lamda = np.roots([G,H,I,J])
    comp = np.iscomplex(lamda)
    lamda = lamda*comp
    lamda0 = 0
    for l in lamda:
        if l != 0:
            lamda0 = np.real(l)
            break
    if lamda0 == 0:
        return P
    
    A = 1 + lamda0
    B = -cos_alpha
    C = (b**2-a**2)/b**2 - lamda0*(c**2/b**2)
    D = -lamda0 * cos_gamma
    E = (a**2/b**2 + lamda0*c**2/b**2)*cos_beta
    F = -a**2/b**2 + lamda0*((b**2 - c**2)/b**2)
    if (B**2-A*C)<0 or (E**2-C*F)<0:
        return P
    
    u = []
    v = []
    p = np.sqrt(B**2 - A*C)
    q = sign(B*E - C*D) * np.sqrt(E**2 - C*F)
    m = np.zeros((2,))
    n = np.zeros((2,))
    A_ = np.zeros((2,))
    B_ = np.zeros((2,))
    C_ = np.zeros((2,))
    m[0] = (-B+p)/C
    m[1] = (-B-p)/C
    n[0] = (-(E-q))/C
    n[1] = (-(E+q))/C
    A_[0] = b**2 - (m[0]**2)*(c**2)
    B_[0] = (c**2)*(cos_beta - n[0])*m[0] - (b**2)*cos_gamma
    C_[0] = -c**2*(n[0]**2) + 2*(c**2) *n[0]*cos_beta+(b**2) - (c**2)
    A_[1] = b**2 - (m[1]**2)*(c**2)
    B_[1] = (c**2)*(cos_beta - n[1])*m[1] - (b**2)*cos_gamma
    C_[1] = -c**2*(n[1]**2) + 2*(c**2)*n[1]*cos_beta+(b**2) - (c**2)
    
    for i in range(2):
        if (B_[i]**2 - A_[i]*C_[i]) >= 0:
            u_large = -sign(B_[i])/A_[i]*(abs(B_[i])+ np.sqrt(B_[i]**2 - A_[i]*C_[i]))
            u_small = C_[i]/(A_[i]*u_large)
            v_large = u_large*m[i] + n[i]
            v_small = u_small*m[i] + n[i]
            u.append(u_large)
            u.append(u_small)
            v.append(v_large)
            v.append(v_small)
            
    if u == []:
        return P
    
    
    lowest_error = np.inf
    for i in range(len(u)):
        s1 = np.sqrt(a**2/(u[i]**2 + v[i]**2 - 2*v[i]*u[i]*cos_alpha))
        s2 = u[i]*s1
        s3 = v[i]*s1
        
        p_cam1 = s1*j1
        p_cam2 = s2*j2
        p_cam3 = s3*j3
        
        p = np.hstack((np.hstack((P1_X,P2_X)),P3_X))
        p_cam = np.hstack((np.hstack((p_cam1,p_cam2)),p_cam3))
        
        mu_p = np.mean(p,axis=1).reshape(3,1)
        mu_p_cam = np.mean(p_cam, axis=1).reshape(3,1)
        
        q_p = p-mu_p
        q_p_cam = p_cam-mu_p_cam
        
        W = q_p_cam.dot(q_p.T)
        U,D,Vh = np.linalg.svd(W)
        if np.linalg.det(U)*np.linalg.det(Vh.T) < 0:
            d = np.eye(3)
            d[2,2] = -1
            R = np.dot(np.dot(U,d), Vh)
        else:
            R = np.dot(U, Vh)

        t = p_cam[:,0:1] - R.dot(p[:,0:1])
        P = np.concatenate((R, t), axis=1)
        
        
        error = 0
        for j in range(x.shape[1]):
            X_cam = K@P@Homogenize(X[:,i:i+1])
            error = error + np.linalg.norm(Dehomogenize(X_cam) - x[:,i:i+1])
            
        if error < lowest_error:
            lowest_error = error
            best_P = P
    
    return best_P

def MSAC(x, X, K, thresh, tol, p):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    K - camera calibration matrix
    #    thresh - cost threshold
    #    tol - reprojection error tolerance 
    #    p - probability that as least one of the random samples does not contain any outliers   
    #
    # Output:
    #    consensus_min_cost - final cost from MSAC
    #    consensus_min_cost_model - camera projection matrix P
    #    inliers - list of indices of the inliers corresponding to input data
    #    trials - number of attempts taken to find consensus set
    
    trials = 0
    max_trials = np.inf
    consensus_min_cost = np.inf
    consensus_min_cost_model = np.zeros((3,4))
    #codimension = 2
    
    while(trials < max_trials):        
        r = random.sample(range(0,x.shape[1]-1), 3)
        p1_x = x[:,r[0]:r[0]+1]
        p2_x = x[:,r[1]:r[1]+1]
        p3_x = x[:,r[2]:r[2]+1]
        P1_X = X[:,r[0]:r[0]+1]
        P2_X = X[:,r[1]:r[1]+1]
        P3_X = X[:,r[2]:r[2]+1]
        
        
        
        model_cur = model(p1_x,p2_x,p3_x,P1_X,P2_X,P3_X,K,x,X)
        if(model_cur.any() == False):
            continue
        cost,error,num_inliers = ComputeCostInlier(model_cur,x,X,K)
        
        #define max trail
        if(cost < consensus_min_cost):
            consensus_min_cost = cost
            consensus_min_cost_model = model_cur
            w = num_inliers/x.shape[1]
            max_trials = np.log(1-p)/np.log(1-w**3)
        
        trials += 1
        print("cost:",cost, "trial:",trials)
        
        
        if(consensus_min_cost < thresh):
            break;
        
    num_inliers = 0
    num_inliers,inliers = inlier(consensus_min_cost_model,x,X,K,tol)
    
    return consensus_min_cost, consensus_min_cost_model, inliers, trials

if __name__ == '__main__':
    # MSAC parameters 
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
    
    # choose just the inliers
    x = x0[:,inliers]
    X = X0[:,inliers]
    
    toc=time.time()
    time_total=toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('%d iterations'%trials)
    print('inlier count: ',len(inliers))
    print('MSAC Cost=%.9f'%cost_MSAC)
    print('P = ')
    print(P_MSAC)
    print('inliers: ',inliers)