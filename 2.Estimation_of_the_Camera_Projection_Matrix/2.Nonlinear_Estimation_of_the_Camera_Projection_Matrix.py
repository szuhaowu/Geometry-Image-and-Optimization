# -*- coding: utf-8 -*-
"""
Nonlinear Estimation of the Camera Projection Matrix (Uncalibrated camera)
@author: Szu-Hao Wu
"""

import numpy as np
import time
from Linear_Estimation_of_the_Camera_Projection_Matrix import Homogenize,Dehomogenize,Normalize,DLT,displayResults,ComputeCost

# Note that np.sinc is different than defined in class
def Sinc(x):
    # Returns a scalar valued sinc value
    """your code here"""
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

    
def Jacobian(P,p,X):
    # compute the jacobian matrix
    #
    # Input:
    #    P - 3x4 projection matrix
    #    p - 11x1 homogeneous parameterization of P
    #    X - 3n 3D scene points
    # Output:
    #    J - 2nx11 jacobian matrix
    J = np.zeros((2*X.shape[1],11))
    
    X_hom = Homogenize(X)
    x_hom = np.dot(P,X_hom)
    x = Dehomogenize(x_hom)
    #zero = np.zeros((1,4))
    for i in range(X.shape[1]):
        w = P[2,:].dot(X_hom[:,i:i+1])[0]
        
        dx_dp = np.zeros((2,12))
        dx_dp[0,0:4] = X_hom[:,i:i+1].T
        dx_dp[0,8:12] = -x[0,i:i+1][0]*X_hom[:,i:i+1].T
        dx_dp[1,4:8] = X_hom[:,i:i+1].T
        dx_dp[1,8:12] = -x[1,i:i+1][0]*X_hom[:,i:i+1].T
        dx_dp = 1/w * dx_dp
        #print(dx_dp)
        
        p_bar = P.reshape(-1,1)
        #a = p_bar[0,0]
        b = p_bar[1:p_bar.shape[0],0]
        da_dp = np.zeros((1,11))
        db_dp = np.zeros((11,11))
        dpbar_dp = np.zeros((12,11))
        if(np.linalg.norm(p) == 0):
            db_dp = (1/2)*np.eye(11)
        else:
            da_dp = -(1/2)*b.reshape((1,11))
            db_dp = Sinc(np.linalg.norm(p)/2)/2*np.eye(p.shape[0])+(dSinc(np.linalg.norm(p)/2)/(4*np.linalg.norm(p)))*p.dot(p.T)
        
        
        dpbar_dp = np.vstack((da_dp, db_dp))
        J[2*i:2*(i+1),:] = dx_dp.dot(dpbar_dp)
        
    return J


def Parameterize(P):
    # wrapper function to interface with LM
    # takes all optimization variables and parameterizes all of them
    # in this case it is just P, but in future assignments it will
    # be more useful
    return ParameterizeHomog(P.reshape(-1,1))


def Deparameterize(p):
    # Deparameterize all optimization variables
    return DeParameterizeHomog(p).reshape(3,4)


def ParameterizeHomog(V):
    # Given a homogeneous vector V return its minimal parameterization
    V = V/np.linalg.norm(V)
    a = V[0,0]
    b = V[1:V.shape[0],:]
    v_hat = (2/(Sinc(np.arccos(a))))*b
    v_hat_norm = np.linalg.norm(v_hat)
    if(v_hat_norm> np.pi):
        v_hat = (1-2*np.pi/v_hat_norm)*np.ceil((v_hat_norm-np.pi)/(2*np.pi))*v_hat
    
    return v_hat

def DeParameterizeHomog(v):
    # Given a parameterized homogeneous vector return its deparameterization
    v_bar = np.zeros((v.shape[0]+1,1))
    v_norm = np.linalg.norm(v)
    a = np.cos(v_norm/2)
    b = (Sinc(v_norm/2)/2)*v.T
    v_bar[0,0] = a
    v_bar[1:v_bar.shape[0],0] = b
    v_bar = v_bar / np.linalg.norm(v_bar)
    
    return v_bar

def Costfromepison(epison,sigmax):
    #return epison.T.dot(np.linalg.inv(sigmax)).dot(epison)
    return np.dot(np.dot(np.transpose(epison), np.linalg.inv(sigmax)), epison)


def LM(P, x, X, max_iters, lam):
    # Input:
    #    P - initial estimate of P
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    # Output:
    #    P - Final P (3x4) obtained after convergence
    
    # data normalization
    x, T = Normalize(x)
    X, U = Normalize(X)
    
    # you may modify this so long as the cost is computed
    # at each iteration
    P_normal = T.dot(P).dot(np.linalg.inv(U))
    x_hat = P_normal.dot(X)
    p_normal = Parameterize(P_normal)
    epison = Dehomogenize(x) - Dehomogenize(x_hat)
    epison = epison.reshape(2*x.shape[1],-1,order = 'F') #epison = 2n*1
    for i in range(max_iters): 
        sigmax = (T[0,0]**2)*np.eye(2*x.shape[1]) #sigmax = 2n*2n
        J = Jacobian(Deparameterize(p_normal),p_normal,Dehomogenize(X)) #J = 2n*11
        
        A = (J.T).dot(np.linalg.inv(sigmax)).dot(J) + lam*np.eye(11) #11*11
        b = (J.T).dot(np.linalg.inv(sigmax)).dot(epison) #11*1
        delta = np.linalg.inv(A).dot(b)
        #normal_matrix = np.dot(np.dot(np.transpose(J), np.linalg.inv(sigmax)), J) + lam*np.eye(11)
        #normal_vector = np.dot(np.dot(np.transpose(J), np.linalg.inv(sigmax)), error)
        #delta = np.dot(np.linalg.inv(normal_matrix), normal_vector)
        
        
        p0 = p_normal + delta
        P0 = Deparameterize(p0)
        x0_hat = P0.dot(X)
        epison0 = Dehomogenize(x) - Dehomogenize(x0_hat)
        epison0 = epison0.reshape(2*x.shape[1],-1,order = 'F')
    
        cost_prev = Costfromepison(epison,sigmax)
        cost0 = Costfromepison(epison0,sigmax)
        if(cost0 > cost_prev):
            lam = 10*lam
            cost = cost_prev
        else:
            cost = cost0
            p_normal = Parameterize(P0)
            epison = epison0
            lam = 0.1*lam
        print ('iter %03d Cost %.9f'%(i+1, cost))
      
    # data denormalization
    P = np.linalg.inv(T) @ P0 @ U
    return P

if __name__ == '__main__':
    # LM hyperparameters
    x=np.loadtxt('points2D.txt').T
    X=np.loadtxt('points3D.txt').T

    # compute the linear estimate without data normalization
    print ('Running DLT without data normalization')
    time_start=time.time()
    P_DLT = DLT(x, X, normalize=False)
    cost = ComputeCost(P_DLT, x, X)
    
    lam = .001
    max_iters = 100
    
    # Run LM initialized by DLT estimate with data normalization
    print ('Running LM with data normalization')
    print ('iter %03d Cost %.9f'%(0, cost))
    time_start=time.time()
    P_LM = LM(P_DLT, x, X, max_iters, lam)
    time_total=time.time()-time_start
    print('took %f secs'%time_total)
    
    # Report your P_LM final value here!
    displayResults(P_LM, x, X, 'P_LM')
