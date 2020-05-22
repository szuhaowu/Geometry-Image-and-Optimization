"""
Estimation of planar projective transformation - Outlier rejection (RANSAC)
@author: Szu-Hao Wu
"""
import numpy as np
from scipy.stats import chi2
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Feature_detection_and_Feature_matching import RunFeatureMatching

def DisplayResults(H, title):
    print(title+' =')
    print (H/np.linalg.norm(H)*np.sign(H[-1,-1]))

def Homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]

def model(x,x_):
    #Use 4-points algorithm to estimate the planar projective transformation from the 2D points in image 1 to the 2D points in image 2
    x_ho = Homogenize(x)
    x__ho = Homogenize(x_)
    temp_x = x_ho[:,0:3]
    temp_x_ = x__ho[:,0:3]
    
    lam_source = np.linalg.inv(temp_x).dot(x_ho[:,3])
    lam_target = np.linalg.inv(temp_x_).dot(x__ho[:,3])
    
    for i in range(3):
        temp_x[:,i] *= lam_source[i]
        temp_x_[:,i] *= lam_target[i]
    
    H = temp_x_.dot(np.linalg.inv(temp_x))
    
    return H
    
def computeSampsonError(match1,match2,model):
    #Compute the Sampson error to calculate the cost of the model(H)
    #Sampson correction algorithm
    sampson_error = np.zeros((match1.shape[1],))     
    for i in range(np.shape(match1)[1]):
        Ah = np.zeros((2,1))
        Ah[0,0] = -(match1[0,i]*model[1,0] + match1[1,i]*model[1,1] + model[1,2]) + match2[1,i]*(match1[0,i]*model[2,0]+match1[1,i]*model[2,1] + model[2,2])
        Ah[1,0] = (match1[0,i]*model[0,0] + match1[1,i]*model[0,1] + model[0,2]) - match2[0,i]*(match1[0,i]*model[2,0]+match1[1,i]*model[2,1] + model[2,2])
        J = np.zeros((2,4))
        J[0,0] = -model[1,0] + match2[1,i]*model[2,0]
        J[0,1] = -model[1,1] + match2[1,i]*model[2,1]
        J[0,3] = match1[0,i]*model[2,0] + match1[1,i]*model[2,1] + model[2,2]
        J[1,0] = model[0,0] - match2[1,i]*model[2,0] 
        J[1,1] = model[0,1] - match2[1,i]*model[2,1] 
        J[1,2] = -(match1[0,i]*model[2,0] + match1[1,i]*model[2,1] + model[2,2])
        sampson_error[i] = np.transpose(Ah)@(np.linalg.inv(J@J.T))@Ah
        
    return sampson_error

def computeCost(n,sampson_error,tol):
    #Compute sampson cost from sampson error
    cost = 0
    num_inliers = 0
    for i in range(n):
        if sampson_error[i] < tol:
            cost = cost + sampson_error[i]
            num_inliers +=1
        else:
            cost = cost + tol
            
    return cost,num_inliers
            
def inlier(consensus_min_cost_model,match1,match2,tol):
    num_inliers = 0
    inliers = []
    for i in range(match1.shape[1]):
        x = consensus_min_cost_model@Homogenize(match1[:,i:i+1])
        x = Dehomogenize(x)
        error = np.linalg.norm(x - match2[:,i:i+1])
        if error < tol:
            inliers.append(i)
            num_inliers += 1
            
            
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
    #    consensus_min_cost_model - planar projective transformation matrix H
    #    inliers - list of indices of the inliers corresponding to input data
    #    trials - number of attempts taken to find consensus set
    
    trials = 0
    max_trials = np.inf
    consensus_min_cost = np.inf
    consensus_min_cost_model = np.zeros((3,4))
    inliers = np.random.randint(0, 200, size=100)
    while(trials < max_trials):   
        r = random.sample(range(0,pts1.shape[1]-1), 4)
        x = np.hstack((pts1[:,r[0]:r[0]+1],pts1[:,r[1]:r[1]+1],pts1[:,r[2]:r[2]+1],pts1[:,r[3]:r[3]+1]))
        x_ = np.hstack((pts2[:,r[0]:r[0]+1],pts2[:,r[1]:r[1]+1],pts2[:,r[2]:r[2]+1],pts2[:,r[3]:r[3]+1]))

        model_cur = model(x,x_)
            
        sampson_error = computeSampsonError(pts1,pts2,model_cur)
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
    num_inliers,inliers = inlier(consensus_min_cost_model,pts1,pts2,tol)
    
    
    return consensus_min_cost, consensus_min_cost_model, inliers, trials


if __name__ == '__main__':
    # MSAC parameters 
    w = 11
    t = 0.87
    d = 0.8
    p = 150
    
    tic = time.time()
    # run the feature matching algorithm on the input images and detected features
    inds,scores = RunFeatureMatching(I1, I2, pts1, pts2, w, t, d, p)
    toc = time.time() - tic
    
    print('took %f secs'%toc)
    
    # create new matrices of points which contain only the matched features 
    match1 = pts1[:,inds[0,:].astype('int')]
    match2 = pts2[:,inds[1,:].astype('int')]
    
    # display the results
    plt.figure(figsize=(14,24))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(I1)
    ax2.imshow(I2)
    plt.title('found %d putative matches'%match1.shape[1])
    for i in range(match1.shape[1]):
        x1,y1 = match1[:,i]
        x2,y2 = match2[:,i]
        ax1.plot([x1, x2],[y1, y2],'-r')
        ax1.add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
        ax2.plot([x2, x1],[y2, y1],'-r')
        ax2.add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))
    
    plt.show()
    
    # test 1-1
    print('unique points in image 1: %d'%np.unique(inds[0,:]).shape[0])
    print('unique points in image 2: %d'%np.unique(inds[1,:]).shape[0])
