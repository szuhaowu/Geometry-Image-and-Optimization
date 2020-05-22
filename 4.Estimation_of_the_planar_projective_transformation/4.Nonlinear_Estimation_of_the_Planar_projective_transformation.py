"""
Nonlinear Estimation of planar projective transformation - LM
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

def Sinc(x):
    # Returns a scalar valued sinc value
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

def Parameterize(P):
    # wrapper function to interface with LM
    # takes all optimization variables and parameterizes all of them
    # in this case it is just P, but in future assignments it will
    # be more useful
    return ParameterizeHomog(P.reshape(-1,1))


def Deparameterize(p):
    # Deparameterize all optimization variables
    return DeParameterizeHomog(p).reshape(3,3)


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


def getScenePoints(x1,x2,H):
    x1_scene = np.zeros((2,x1.shape[1]))
    for i in range(np.shape(x1)[1]):
        Ah = np.zeros((2,1))
        Ah[0,0] = -(x1[0,i]*H[1,0] + x1[1,i]*H[1,1] + H[1,2]) + x2[1,i]*(x1[0,i]*H[2,0]+x1[1,i]*H[2,1] + H[2,2])
        Ah[1,0] = (x1[0,i]*H[0,0] + x1[1,i]*H[0,1] + H[0,2]) - x2[0,i]*(x1[0,i]*H[2,0]+x1[1,i]*H[2,1] + H[2,2])
        J = np.zeros((2,4))
        J[0,0] = -H[1,0] + x2[1,i]*H[1,0]
        J[0,1] = -H[1,1] + x2[1,i]*H[2,1]
        J[0,3] = x1[0,i]*H[2,0] + x1[1,i]*H[2,1] + H[2,2]
        J[1,0] = H[0,0] - x2[1,i]*H[2,0] 
        J[1,1] = H[0,1] - x2[1,i]*H[2,1] 
        J[1,2] = -(x1[0,i]*H[2,0] + x1[1,i]*H[2,1] + H[2,2])
        
        lamb = -np.linalg.inv(J.dot(J.T)).dot(Ah) #Ah = epsilon1
        delta = J.T.dot(lamb) #delta = 4*1 [x1x,x1y,x2x,x2y]
        #x1_scene = np.zeros((2,1))
        x1_scene[0,i] = x1[0,i] + delta[0,0]
        x1_scene[1,i] = x1[1,i] + delta[1,0]
    
    return x1_scene

def getA(x_scene,H):
    H_vec = np.reshape(H,(-1,1),order='C')
    h = ParameterizeHomog(H_vec)
    h_norm = np.linalg.norm(h)
    x_scene_paho = DeParameterizeHomog(x_scene)
    x2_paho = H@x_scene_paho
    x2_pa = Dehomogenize(x2_paho)
    h_norm = np.linalg.norm(h)
    A__temp = np.zeros((2,8))
    
    #A  2*9 dot 9*8 = 2*8
    #dx_scene_bar/dh 2*9
    dx_scene_bar_dh = np.zeros((2,9))
    w_ = H[2:3,:].dot(x_scene_paho)
    dx_scene_bar_dh[0:1,:] = np.hstack((x_scene_paho.T,np.zeros((1,3)),(-x2_pa[0,0])*x_scene_paho.T))
    dx_scene_bar_dh[1:2,:] = np.hstack((np.zeros((1,3)),x_scene_paho.T,(-x2_pa[1,0])*x_scene_paho.T))
    dx_scene_bar_dh = dx_scene_bar_dh/w_
    
    #dh_bar/dh 9*8
    a = H_vec[0,0]
    b = H_vec[1:,0]
    da_dh = np.zeros((1,8))
    db_dh = np.zeros((8,8))
    #dhbar_dh = np.zeros((9,8))
    if(h_norm == 0):
        db_dh = (1/2)*np.eye(8)
    else:
        da_dh = -(1/2)*b.reshape((1,8))
        db_dh = Sinc(h_norm/2)/2*np.eye(h.shape[0])+(dSinc(h_norm/2)/(4*h_norm))*h.dot(h.T)
    dhbar_dh = np.vstack((da_dh, db_dh))
    A__temp = dx_scene_bar_dh@dhbar_dh
    
    return A__temp
    
def getB(x_scene,H):
    B_temp = np.zeros((2,2))
    x_scene_ho = Homogenize(x_scene).reshape(3,1)
    x_scene_paho = ParameterizeHomog(Homogenize(x_scene)).reshape(2,1)
    x_norm = np.linalg.norm(x_scene_ho)
    w_ = H[2:3,:].dot(x_scene_ho)
    #B 2*3 dot 3*2 = 2*2
    #dx_scene/dx_scene_bar 2*3
    dx_scene_dx = np.zeros((2,3))
    dx_scene_dx[0:1,:] = H[0:1,:] - x_scene[0,0]*H[2:3,:]
    dx_scene_dx[1:2,:] = H[1:2,:] - x_scene[1,0]*H[2:3,:]
    dx_scene_dx = dx_scene_dx/w_
    
    #dx_scene_bar/dx 3*2
    a = x_scene_ho[0,0]
    b = x_scene_ho[1:,0]
    da_dx = np.zeros((1,2))
    db_dx = np.zeros((2,2))
    #dx_scene_bar_dx = np.zeros((3,2))
    
    if(x_norm == 0):
        db_dx = (1/2)*np.eye(2)
    else:
        da_dx = -(1/2)*b.reshape((1,2))
        db_dx = Sinc(x_norm/2)/2*np.eye(2 )+(dSinc(x_norm/2)/(4*x_norm))*x_scene_paho.dot(x_scene_paho.T)
    dx_scene_bar_dx = np.vstack((da_dx, db_dx))
    B_temp = dx_scene_dx@dx_scene_bar_dx
    return B_temp

def Jacobian(x_scene,H):
    #Compute the sparse Jacobian matrix
    A_ = []
    B = []
    B_ = []
    n = x_scene.shape[1]   
    #x_scene_paho = DeParameterizeHomog(x_scene)
    
    x2_ho = H@Homogenize(x_scene)
    x2 = Dehomogenize(x2_ho)

    #print(x_scene.shape)
    for i in range(n):
        A__temp = np.zeros((2,8))
        B_temp = np.zeros((2,2))
        B__temp = np.zeros((2,2))
        
        
        A__temp = getA(x_scene[:,i:i+1],H)
        B_temp = getB(x_scene[:,i:i+1],np.eye(3))
        B__temp = getB(x2[:,i:i+1],H)
        
        B.append(B_temp)
        A_.append(A__temp)
        B_.append(B__temp)
        
    return A_,B,B_

def LM(H, x1, x2, max_iters, lam):
    # Input:
    #    H - DLT estimate of planar projective transformation matrix
    #    x1 - inhomogeneous inlier points in image 1
    #    x2 - inhomogeneous inlier points in image 2
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    # Output:
    #    H - Final H (3x3) obtained after convergence
    
    # data normalization
    x1, T1 = Normalize(x1)
    x2, T2 = Normalize(x2)
    x_scene = getScenePoints(x1,x2,H)
    n = x1.shape[1]
    cov1 = (T1[0,0]**2) * np.eye(2)
    cov2 = (T2[0,0]**2) * np.eye(2)  
    U_ = np.zeros((8,8))
    V = np.zeros((2*n,2))
    W = np.zeros((8*n,2))
    ea = np.zeros((8,1))
    eb = np.zeros((2*n,1))
    s_aug = np.zeros((8,8))
    e_aug = np.zeros((8,1))
    cost = 0
    for k in range(max_iters): 
        A_,B,B_ = Jacobian(x_scene,H)
        h = Parameterize(H)
        for i in range(n):
            A__temp = A_[i]
            B_temp = B[i]
            B__temp = B_[i]
            #print(A__temp)
            U_ += np.transpose(A__temp)@np.linalg.inv(cov2)@A__temp
            V[2*i:2*i+2,:] = B_temp.T@np.linalg.inv(cov1)@B_temp + B__temp.T@np.linalg.inv(cov1)@B__temp
            W[8*i:8*i+8,:] = A__temp.T@np.linalg.inv(cov2)@B_temp
            
            #normal equation
            epsilon_i = (Dehomogenize(x1[:,i:i+1]) -  x_scene[:,i:i+1]) #problem
            epsilon_i_ = (Dehomogenize(x2[:,i:i+1]) -  Dehomogenize(H@Homogenize(x_scene[:,i:i+1]))) 
            ea += A__temp.T@np.linalg.inv(cov2)@epsilon_i
            eb[2*i:2*i+2,:] = B_temp.T@np.linalg.inv(cov1)@epsilon_i+ B__temp.T@np.linalg.inv(cov2)@epsilon_i_
            s_aug += W[8*i:8*i+8,:]@np.linalg.inv(lam*V[2*i:2*i+2,:])@np.transpose(W[8*i:8*i+8,:])
            e_aug += W[8*i:8*i+8,:]@np.linalg.inv(lam*V[2*i:2*i+2,:])@eb[2*i:2*i+2,:]
            
        s_ = lam*U_ - s_aug
        e_ = ea - e_aug
        delta_a = np.linalg.inv(s_)@e_ #8*1
        delta_b = np.zeros((2*n,1)) 
        for i in range(n):
            delta_b[2*i:2*i+2,:] = np.linalg.inv(lam*V[2*i:2*i+2,:])@(eb[2*i:2*i+2,:]-W[8*i:8*i+8,:].T@delta_a)
        
        h0 = h+delta_a
        H0 = Deparameterize(h0)
        x_scene0 = np.zeros(x_scene.shape)
        for i in range(n):
            x_scene0[:,i:i+1] = x_scene[:,i:i+1]+delta_b[2*i:2*i+2,:]
        
        cost_prev = computeSampsonCost(x1,x2,H,T1,T2)
        cost0 = computeSampsonCost(x1,x2,H0,T1,T2)
        if(cost0 >= cost_prev):
            lam = 10*lam
        else:
            cost = cost0
            H = H0
            x_scene = x_scene0
            lam = 0.1*lam
            print ('iter %03d Cost %.9f'%(k+1, cost))
    
    return H

if __name__ == '__main__':
    x1 = xin1 #inlier 2D points from outlier rejection
    x2 = xin2
    
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    # Run LM initialized by DLT estimate with data normalization
    print ('Running sparse LM with data normalization')
    print ('iter %03d Cost %.9f'%(0, cost))
    time_start=time.time()
    H_LM = LM(H_DLT, x1, x2, max_iters, lam) #H_DLT from linear estimation of planar projective transformation
    time_total=time.time()-time_start
    print('took %f secs'%time_total)
    
    # display your converged H_LM, scaled with its frobenius norm
    DisplayResults(H_LM, 'H_LM')