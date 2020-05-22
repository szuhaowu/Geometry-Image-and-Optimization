"""
Estimation of the fundamental matrix - Outlier rejection (RANSAC)
@author: Szu-Hao Wu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sympy import Matrix, solve ,symbols
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

def model(x,x_):
    #Use the 7-point algorithm to estimate the fundamental matrix, resulting in 1 or 3 solutions.
    A = np.zeros((7,9))
    for i in range(7):
        A[i:i+1, :] = np.kron(x_[:,i:i+1].T, x[:,i:i+1].T)
    U,D,V = np.linalg.svd(A)
    a = V[8:9,:]
    b = V[7:8,:]
    F1 = np.reshape(a, (3,3))
    F2 = np.reshape(b, (3,3))
    alpha = symbols('alpha')
    F = Matrix(alpha*F1 + F2)
    answer = solve(Matrix.det(F), alpha)
    alpha = np.array(answer).astype(np.complex64)
    alpha = np.real(alpha)
    
    F_best = np.zeros((3,3))
    error_best = np.inf
    for i in range(3):
        F_temp = alpha[i]*F1 + F2
        error = 0
        for j in range(x.shape[1]):
            numerator = (x_[:,j:j+1].T@F_temp@x[:,j:j+1])**2
            denominator = (x_[:,j:j+1].T@F_temp[:,0:1])**2 + (x_[:,j:j+1].T@F_temp[:,1:2])**2 + (F_temp[0:1,:]@x[:,j:j+1])**2 + (F_temp[1:2,:]@x[:,j:j+1])**2
            error += numerator/denominator
        if error_best > error:
            error_best = error
            F_best = F_temp
    
    
    return F_best
    
def computeSampsonError(pts1,pts2,model):
    sampson_error = np.zeros((pts1.shape[1],))     
    for j in range(pts1.shape[1]):
        numerator = (pts2[:,j:j+1].T@model@pts1[:,j:j+1])**2
        denominator = (pts2[:,j:j+1].T@model[:,0:1])**2 + (pts2[:,j:j+1].T@model[:,1:2])**2 + (model[0:1,:]@pts1[:,j:j+1])**2 + (model[1:2,:]@pts1[:,j:j+1])**2
        sampson_error[j] = numerator/denominator
         
    return sampson_error

def computeCost(n,sampson_error,tol):
    cost = 0
    num_inliers = 0
    for i in range(n):
        if sampson_error[i] < tol:
            cost = cost + sampson_error[i]
            num_inliers +=1
        else:
            cost = cost + tol
            
    return cost,num_inliers
            
def inlier(model,pts1,pts2,tol):
    num_inliers = 0
    inliers = []
    outliers = []
    for j in range(pts1.shape[1]):
        numerator = (pts2[:,j:j+1].T@model@pts1[:,j:j+1])**2
        denominator = (pts2[:,j:j+1].T@model[:,0:1])**2 + (pts2[:,j:j+1].T@model[:,1:2])**2 + (model[0:1,:]@pts1[:,j:j+1])**2 + (model[1:2,:]@pts1[:,j:j+1])**2
        sampson_error = numerator/denominator
        if sampson_error < tol:
            inliers.append(j)
            num_inliers += 1
        else:
            outliers.append(j)
            
    return num_inliers,inliers

def MSAC(pts1, pts2, thresh, tol, p):
    # Inputs:
    #    pts1 - matched feature correspondences in image 1
    #    pts2 - matched feature correspondences in image 2
    #    thresh - cost threshold
    #    tol - reprojection error tolerance 
    #    p - probability that as least one of the random samples does not contain any outliers   
    #
    # Output:
    #    consensus_min_cost - final cost from MSAC
    #    consensus_min_cost_model - fundamental matrix F
    #    inliers - list of indices of the inliers corresponding to input data
    #    trials - number of attempts taken to find consensus set
    
    trials = 0
    max_trials = np.inf
    consensus_min_cost = np.inf
    consensus_min_cost_model = np.zeros((3,4))
    inliers = np.random.randint(0, 200, size=100)
    pts1_ho = Homogenize(pts1)
    pts2_ho = Homogenize(pts2)
    
    while(trials < max_trials):   
        r = random.sample(range(0,pts1.shape[1]-1), 7)
        
        x = pts1_ho[:,r]
        x_ = pts2_ho[:,r]
        
        model_cur = model(x,x_)
            
        sampson_error = computeSampsonError(pts1_ho,pts2_ho,model_cur)
        cost,num_inliers = computeCost(pts1.shape[1],sampson_error,tol)
        #define max trail
        if(cost < consensus_min_cost):
            consensus_min_cost = cost
            consensus_min_cost_model = model_cur
            w = num_inliers/pts1.shape[1]
            max_trials = np.log(1-p)/np.log(1-w**3)
            #print('c',max_trials)
        
        trials += 1
        print("cost:",cost, "trial:",trials)
        
        if(consensus_min_cost < thresh):
            break;
        
    num_inliers = 0
    num_inliers,inliers= inlier(consensus_min_cost_model,pts1_ho,pts2_ho,tol)
    random_seed = 85
    
    return consensus_min_cost, consensus_min_cost_model, inliers, trials, random_seed

if __name__ == '__main__':
    # MSAC parameters 
    thresh = 100
    tol = 3
    p = 0.99
    alpha = 0.95
    
    tic=time.time()
    
    cost_MSAC, F_MSAC, inliers, trials, random_seed= MSAC(match1, match2, thresh, tol, p)
    
    # choose just the inliers
    xin1 = match1[:,inliers] #matching 2D points from features matching
    xin2 = match2[:,inliers]
    outliers = np.setdiff1d(np.arange(pts1.shape[1]),inliers)
    
    toc=time.time()
    time_total=toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('%d iterations'%trials)
    print('inlier count: ',len(inliers))
    print('inliers: ',inliers)
    print('MSAC Cost = %.9f'%cost_MSAC)
    print('random_seed',random_seed)
    DisplayResults(F_MSAC, 'F_MSAC')
    
    # display the figures
    plt.figure(figsize=(14,8))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(I1)
    ax2.imshow(I2)
    plt.title('found %d inliers'%xin1.shape[1])
    
    for i in range(xin1.shape[1]):
        x1,y1 = xin1[:,i]
        x2,y2 = xin2[:,i]
        ax1.plot([x1, x2],[y1, y2],'-r')
        ax1.add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
        ax2.plot([x2, x1],[y2, y1],'-r')
        ax2.add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))
    
    plt.show()