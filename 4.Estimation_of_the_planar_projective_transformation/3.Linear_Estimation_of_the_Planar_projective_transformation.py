"""
Linear Estimation of planar projective transformation - DLT
@author: Szu-Hao Wu
"""
import numpy as np
import time

def DisplayResults(H, title):
    print(title+' =')
    print (H/np.linalg.norm(H)*np.sign(H[-1,-1]))
    
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

def leftNullSpace(x):
    n = x.shape[0]
    e1 = np.zeros((n,1))
    e1[0] = 1
    x_norm = np.linalg.norm(x)
    v = x + np.sign(x[0])*x_norm*e1
    Hv = np.eye(n)-2*(v.dot(v.T))/(np.dot(v.T,v))
    
    return Hv[1:,:]

def computeSampsonCost(x1,x2,H,T1,T2):  
    #compute Sampson Cost to calculate the cost in DLT process
    cost = 0
    for i in range(np.shape(x1)[1]):
        Ah = np.zeros((2,1))
        Ah[0,0] = -(x1[0,i]*H[1,0] + x1[1,i]*H[1,1] + H[1,2]) + x2[1,i]*(x1[0,i]*H[2,0]+x1[1,i]*H[2,1] + H[2,2])
        Ah[1,0] = (x1[0,i]*H[0,0] + x1[1,i]*H[0,1] + H[0,2]) - x2[0,i]*(x1[0,i]*H[2,0]+x1[1,i]*H[2,1] + H[2,2])
        J = np.zeros((2,4))
        J[0,0] = -H[1,0] + x2[1,i]*H[2,0]
        J[0,1] = -H[1,1] + x2[1,i]*H[2,1]
        J[0,3] = x1[0,i]*H[2,0] + x1[1,i]*H[2,1] + H[2,2]
        J[1,0] = H[0,0] - x2[1,i]*H[2,0] 
        J[1,1] = H[0,1] - x2[1,i]*H[2,1] 
        J[1,2] = -(x1[0,i]*H[2,0] + x1[1,i]*H[2,1] + H[2,2])
        
        lamb = np.linalg.inv(J.dot(J.T)).dot(-Ah) 
        delta = J.T.dot(lamb) #delta = 4*1 [x1x,x1y,x2x,x2y]
        x1_scene = np.zeros((2,1))
        x1_scene[0,0] = x1[0,i] + delta[0,0]
        x1_scene[1,0] = x1[1,i] + delta[1,0]
        x2_pro = Dehomogenize(H.dot(Homogenize(x1_scene))).reshape(2,1)
        epsilon1 = delta[0:2,:]
        epsilon2 = x2_pro - Dehomogenize(x2[:,i:i+1])
                       
        cov1 = (T1[0,0]**2) * np.eye(2)
        cov2 = (T2[0,0]**2) * np.eye(2)               
        cost += epsilon1.T.dot(np.linalg.inv(cov1)).dot(epsilon1) + epsilon2.T.dot(np.linalg.inv(cov2)).dot(epsilon2)
        
    return cost

def DLT(x1, x2, normalize=True):
    # Inputs:
    #    x1 - inhomogeneous inlier correspondences in image 1
    #    x2 - inhomogeneous inlier correspondences in image 1
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    H - the DLT estimate of the planar projective transformation   
    #    cost - linear estimate cost
    
    # data normalization
    if normalize:
        print('normalize')
        x1, T1 = Normalize(x1)
        x2, T2 = Normalize(x2)
    # data denormalize
    else:
        x1 = Homogenize(x1)
        x2 = Homogenize(x2)
        T1 = np.eye(3)
        T2 = np.eye(3)
          
    H = np.eye(3)
    A = np.zeros((2*(x1.shape[1]),9))
    for i in range(x1.shape[1]):
        x2_left = leftNullSpace(x2[:,i:i+1])
        A[2*i:2*(i+1),:] = np.kron(x2_left, np.transpose(x1[:,i:i+1]))
    
    u,s,v = np.linalg.svd(A)
    H = v[8:9,:].reshape(H.shape)
    
    #cost = computeCost(H, x1, x2, T1, T2, normalize=True)
    cost = computeSampsonCost(x1,x2,H,T1,T2) 
    # data denormalize
    if normalize:
        print('denormalize')
        #H = np.linalg.inv(T2) @ H @ T1
     
    
    return H, cost

if __name__ == '__main__':
    x1 = xin1 #inlier 2D points from outlier rejection
    x2 = xin2
    # compute the linear estimate without data normalization
    print ('Running DLT without data normalization')
    time_start=time.time()
    H_DLT, cost = DLT(x1, x2, normalize=False)
    DisplayResults(H_DLT, 'without data normalization')
    time_total=time.time()-time_start
    # display the results
    print('took %f secs'%time_total)
    print('Cost=%.9f'%cost)
    
    
    # compute the linear estimate with data normalization
    print ('Running DLT with data normalization')
    time_start=time.time()
    H_DLT, cost = DLT(x1, x2, normalize=True)
    DisplayResults(H_DLT, 'with data normalization')
    time_total=time.time()-time_start
    # display the results
    print('took %f secs'%time_total)
    print('Cost=%.9f'%cost)
    
    # display your H_DLT, scaled with its frobenius norm
    DisplayResults(H_DLT, 'H_DLT')