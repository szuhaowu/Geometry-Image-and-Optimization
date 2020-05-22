"""
Nonlinear Estimation of the fundamental matrix - LM & triangulation
@author: Szu-Hao Wu
"""
import numpy as np
import time
from sympy import symbols ,solveset

#util function
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

#%%LM
    
def projectionMatrix_p(F):
    U,D,V = np.linalg.svd(F)
    
    D[2] = (D[0]+D[1])/2
    D_p = np.diag(D)
    W = np.zeros((3,3))
    W[0,1] = 1
    W[1,0] = -1
    
    Z = np.zeros((3,3))
    Z[0,1] = -1
    Z[1,0] = 1
    Z[2,2] = 1
    
    S = U@W@U.T
    M = U@Z@D_p@V
    e = np.zeros((3,1))
    e[0,0] = S[2,1]
    e[1,0] = S[0,2]
    e[2,0] = S[1,0]
    P = np.hstack((M,e))
    return P

def FundamentalMatrixFromP(P):
    e = P_p[:,3:4]
    e_matrix = np.zeros((3,3))
    e_matrix[2,1] = e[0,0]
    e_matrix[1,2] = -e[0,0]
    e_matrix[0,2] = e[1,0]
    e_matrix[2,0] = -e[1,0]
    e_matrix[1,0] = e[2,0]
    e_matrix[0,1] = -e[2,0]
    F = e_matrix@P_p[0:3,0:3]
    
    return F

def triangulation(x1,x2,F,P_p):
    #use triangulation to compute 3D scene points of 2 corresponding images
    # Input:
    #    x1 - inhomogeneous inlier points in image 1
    #    x2 - inhomogeneous inlier points in image 2
    #    F  - estimated fundamental from camera projection matrix P
    #    P_p - estimated camera projection matrix of camera2
    # Output:
    #    x_scene - 3D scene points
    x1 = Homogenize(x1)
    x2 = Homogenize(x2)
    x_scene = np.zeros((4,x1.shape[1]))
    
    for i in range(x1.shape[1]):
        #two view corrected point
        T = np.zeros((3,3))
        T[0,0] = x1[:,i:i+1][2,0]
        T[1,1] = x1[:,i:i+1][2,0]
        T[2,2] = x1[:,i:i+1][2,0]
        T[0,2] = -x1[:,i:i+1][0,0]
        T[1,2] = -x1[:,i:i+1][1,0]
        T_p = np.zeros((3,3))
        T_p[0,0] = x2[:,i:i+1][2,0]
        T_p[1,1] = x2[:,i:i+1][2,0]
        T_p[2,2] = x2[:,i:i+1][2,0]
        T_p[0,2] = -x2[:,i:i+1][0,0]
        T_p[1,2] = -x2[:,i:i+1][1,0]
        
        #compute e and e_p
        Fs = np.linalg.inv(T_p).T@F@np.linalg.inv(T)
        U,S,V = np.linalg.svd(Fs)
        e = V[2:3,:].T
        U,S,V = np.linalg.svd(Fs.T)
        e_p = V[2:3,:].T
        
        e = np.sqrt(1/(e[0,0]**2 + e[1,0]**2))*e
        e_p = np.sqrt(1/(e_p[0,0]**2 + e_p[1,0]**2))*e_p
        
        #compute rotation matrix
        R = np.zeros((3,3))
        R[0,0] = e[0,0]
        R[1,1] = e[0,0]
        R[2,2] = 1
        R[0,1] = e[1,0]
        R[1,0] = -e[1,0]
        R_p = np.zeros((3,3))
        R_p[0,0] = e_p[0,0]
        R_p[1,1] = e_p[0,0]
        R_p[2,2] = 1
        R_p[0,1] = e_p[1,0]
        R_p[1,0] = -e_p[1,0]
        
        Fs = R_p@Fs@R.T
        f = e[2,0]
        f_p = e_p[2,0]
        a = Fs[1,1]
        b = Fs[1,2]
        c = Fs[2,1]
        d = Fs[2,2]
        
        #solve t
        t = symbols('t') 
        g = list(solveset(t*((a*t+b)**2 + f_p**2*(c*t+d)**2)**2 - (a*d-b*c)*(1+f**2*t**2)**2*(a*t+b)*(c*t+d),t))
        answer = np.array(g).astype(np.complex64)
        answer = np.real(answer)
        cost = np.inf
        
        t_final = 0
        for j in range(len(answer)):
            St = answer[j]**2/(1+f**2*answer[j]**2) + (c*answer[j]+d)**2/((a*answer[j]+b)**2+f_p**2*(c*answer[j]+d)**2)
            if St < cost:
                cost = St
                t_final = answer[j]
        St = 1/f**2 + c**2/(a**2 + f_p**2*c**2)
        if St < cost:
            cost = St
            t_final = float("inf")
        l = np.zeros((3,1))
        l_p = np.zeros((3,1))
        if t_final == float("inf"):
            l[0,0] = f
            l[2,0] = -1
            l_p[0,0] = -f_p*c
            l_p[1,0] = a
            l_p[2,0] = c
        else:
            l[0,0] = t_final*f
            l[1,0] = 1
            l[2,0] = -t_final
            l_p[0,0] = -f_p*(c*t_final + d)
            l_p[1,0] = a*t_final + b
            l_p[2,0] = c*t_final + d
        
        x1_pro = np.zeros((3,1))
        x2_pro = np.zeros((3,1))
        #x = (-ac, -bc, a^2+b^2).T
        x1_pro[0,0] = -l[0,0]*l[2,0] 
        x1_pro[1,0] = -l[1,0]*l[2,0]
        x1_pro[2,0] = l[0,0]**2 + l[1,0]**2
        x2_pro[0,0] = -l_p[0,0]*l_p[2,0]
        x2_pro[1,0] = -l_p[1,0]*l_p[2,0]
        x2_pro[2,0] = l_p[0,0]**2 + l_p[1,0]**2
        
        x_hat = np.linalg.inv(T)@R.T@x1_pro.reshape((3,1))  #reproject to original coordinate      
        x_p_hat = np.linalg.inv(T_p)@R_p.T@x2_pro.reshape((3,1))
        
        #triangulation
        l_p = F@x_hat
        l_p_orth = np.zeros((3,1))
        l_p_orth[0,0] = -l[1,0]*x_p_hat[2,0]
        l_p_orth[1,0] = l[0,0]*x_p_hat[2,0]
        l_p_orth[2,0] = l[1,0]*x_p_hat[0,0] - l[0,0]*x_p_hat[1,0]
        
        plane = P_p.T@l_p_orth
        x_pi = np.zeros((4,1))
        x_pi[0,0] = plane[3,0]*x_hat[0,0]
        x_pi[1,0] = plane[3,0]*x_hat[1,0]
        x_pi[2,0] = plane[3,0]*x_hat[2,0]
        x_pi[3,0] = -plane[0:3,:].T@x_hat
        x_scene[:,i:i+1] = x_pi 
    
    return x_scene      
  
#%%Function for computing sparse Jacobian matrix      
    
def getA(x,P_p,x_scene):
    #Input
    #x_scene => Homogeneous 3D scene points 4*1
    #x_corrected => Homogeneous corrected point 3*1
    #P_p => Homogeneize projection matrix 3*4
    
    x = Dehomogenize(x) #x_corrected => inHomogeneous corrected point 2*1
    #############
    p1T = P_p[0:1,:]
    p2T = P_p[1:2,:]
    p3T = P_p[2:3,:]
    w = p3T@x_scene
    dx_dp = np.zeros((2,12))
    dx_dp[0,:] = np.hstack((x_scene.T,np.zeros((1,4)),(-x[0])*x_scene.T))
    dx_dp[1,:] = np.hstack((np.zeros((1,4)),x_scene.T,(-x[1])*x_scene.T))
    dx_dp = dx_dp / w[0][0]
    
    #############

    #dp_bar/dp 12*11
    p_bar = np.reshape(P_p,(-1,1),order='C')
    p = Parameterize(P_p)
    p_norm = np.linalg.norm(p)
    
    a = p_bar[0,0]
    b = p_bar[1:,:]
    da_dp = np.zeros((1,11))
    db_dp = np.zeros((11,11))
    #dpbar_dp = np.zeros((12,11))
    if(p_norm == 0):
        db_dp = (1/2)*np.eye(11)
    else:
        da_dp = -(1/2)*b.T
        db_dp = Sinc(p_norm/2)/2*np.eye(p.shape[0])+(dSinc(p_norm/2)/(4*p_norm))*p.dot(p.T)
    dpbar_dp = np.vstack((da_dp, db_dp))
    A_p_temp = dx_dp@dpbar_dp
    
    return A_p_temp
    
def getB(x,P,x_scene):
    #x_scene => Homogeneous 3D scene points 4*1
    #x_corrected => Homogeneous corrected point 3*1
    #P => Homogeneize projection matrix 3*4
    
    B_temp = np.zeros((2,3))
    x = Dehomogenize(x)
    ########
    dx_dX = np.zeros((2,4))
    p1T = P[0:1,:]
    p2T = P[1:2,:]
    p3T = P[2:3,:]
    w = p3T@x_scene
    
    dx_dX[0,:] = p1T - x[0]*p3T
    dx_dX[1,:] = p2T - x[1]*p3T
    dx_dX = dx_dX/w[0][0]
    ########
    
    x_scene_pa = Parameterize(x_scene)
    #x_scene = DeParameterizeHomog(x_scene_pa)
    x_norm = np.linalg.norm(x_scene_pa)
    a = x_scene[0,0]
    b = x_scene[1:,:]
    da_dx = np.zeros((1,3))
    db_dx = np.zeros((3,3))
    
    if(x_norm == 0):
        db_dx = (1/2)*np.eye(3)
    else:
        da_dx = -(1/2)*b.T
        db_dx = Sinc(x_norm/2)/2*np.eye(3)+(dSinc(x_norm/2)/(4*x_norm))*x_scene_pa.dot(x_scene_pa.T)
    dx_scene_bar_dx = np.vstack((da_dx, db_dx))
    B_temp = dx_dX@dx_scene_bar_dx
    return B_temp

def Jacobian(x1,x2,P_p,x_scene):
    #Input
    #x_scene => Homogeneous 3D scene points 4*n
    #x1 => Homogeneous corrected point 3*n
    #x2 => Homogeneous corrected point 3*n
    #P_p => Homogeneize projection matrix 3*4
    
    A_p = []
    B = []
    B_p = []
    n = x1.shape[1]   

    for i in range(n):
        A_p_temp = np.zeros((2,11))
        B_temp = np.zeros((2,3))
        B_p_temp = np.zeros((2,3))
        
        A_p_temp = getA(x2[:,i:i+1],P_p,x_scene[:,i:i+1])
        B_temp = getB(x1[:,i:i+1],np.hstack((np.eye(3),np.zeros((3,1)))),x_scene[:,i:i+1])
        B_p_temp = getB(x2[:,i:i+1],P_p,x_scene[:,i:i+1])
        
        A_p.append(A_p_temp)
        B.append(B_temp)
        B_p.append(B_p_temp)
        
    return A_p,B,B_p

#%%LM

def computeCost(epsilon_i,epsilon_i_p,cov1,cov2):
    #Compute cost of LM process
    return epsilon_i.T@np.linalg.inv(cov1)@epsilon_i + epsilon_i_p.T@np.linalg.inv(cov2)@epsilon_i_p

def LM(F, x1, x2, max_iters, lam, x_scene, P_p):
    # Input:
    #    F - DLT estimate of the fundamental matrix
    #    x1 - inhomogeneous inlier points in image 1
    #    x2 - inhomogeneous inlier points in image 2
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    # Output:
    #    F - Final fundamental matrix obtained after convergence
    
    n = x1.shape[1]
    P = np.hstack((np.eye(3),np.zeros((3,1))))

    print('Normalization')
    x1, T1 = Normalize(x1)
    x2, T2 = Normalize(x2)
    x_scene,U = Normalize(Dehomogenize(x_scene)) 
    P = T1@P@np.linalg.inv(U)
    P_p = T2@P_p@np.linalg.inv(U)
    
    cov1 = (T1[0,0]**2) * np.eye(2)
    cov2 = (T2[0,0]**2) * np.eye(2)
    
    cost = 0
    for i in range(n):
        #compute cost
        epsilon_i = (Dehomogenize(x1[:,i:i+1]) -  Dehomogenize(P@x_scene[:,i:i+1])) 
        epsilon_i_p = (Dehomogenize(x2[:,i:i+1]) -  Dehomogenize(P_p@x_scene[:,i:i+1]))
        cost += computeCost(epsilon_i,epsilon_i_p,cov1,cov2)
    print ('start optimaization with cost %.9f'%(cost))
    
    for k in range(max_iters): 
        U_p = np.zeros((11,11))
        V = np.zeros((3*n,3))
        W = np.zeros((11*n,3))
        ea = np.zeros((11,1))
        eb = np.zeros((3*n,1))
        s_aug = np.zeros((11,11))
        e_aug = np.zeros((11,1))
        
        x1_corrected = P@x_scene
        x2_corrected = P_p@x_scene
        A_p,B,B_p = Jacobian(x1_corrected,x2_corrected,P_p,x_scene)
        for i in range(n):
            A_p_temp = A_p[i]
            B_temp = B[i]
            B_p_temp = B_p[i]
            #print(cov1)
            U_p += np.transpose(A_p_temp)@np.linalg.inv(cov2)@A_p_temp
            V[3*i:3*i+3,:] = B_temp.T@np.linalg.inv(cov1)@B_temp + B_p_temp.T@np.linalg.inv(cov2)@B_p_temp
            W[11*i:11*i+11,:] = A_p_temp.T@np.linalg.inv(cov2)@B_p_temp
            #print(W[11*i:11*i+11,:])
            #normal equation
            epsilon_i = (Dehomogenize(x1[:,i:i+1]) -  Dehomogenize(P@x_scene[:,i:i+1])) 
            epsilon_i_p = (Dehomogenize(x2[:,i:i+1]) -  Dehomogenize(P_p@x_scene[:,i:i+1]))
            
            ea += A_p_temp.T@np.linalg.inv(cov2)@epsilon_i_p
            eb[3*i:3*i+3,:] = B_temp.T@np.linalg.inv(cov1)@epsilon_i+ B_p_temp.T@np.linalg.inv(cov2)@epsilon_i_p
            s_aug += W[11*i:11*i+11,:]@np.linalg.inv(lam*np.eye(3)+V[3*i:3*i+3,:])@W[11*i:11*i+11,:].T
            e_aug += W[11*i:11*i+11,:]@np.linalg.inv(lam*np.eye(3)+V[3*i:3*i+3,:])@eb[3*i:3*i+3,:]
      

        s_p = lam*np.eye(11)+U_p - s_aug
        e_p = ea - e_aug
        ########
        #delta_a for p from P_p
        #delta_b for parameterized x_scene
        
        delta_a = np.linalg.inv(s_p)@e_p #11*1
        p = Parameterize(P_p)
        p += delta_a
        P_p_temp = Deparameterize(p)
        
        delta_b = np.zeros((3,1)) 
        
        x_scene_temp = np.zeros(x_scene.shape)
        cost_temp = 0
        for i in range(n):
            delta_b = np.linalg.inv(lam*np.eye(3)+V[3*i:3*i+3,:])@(eb[3*i:3*i+3,:]-W[11*i:11*i+11,:].T@delta_a)
            x_scene_pa = Parameterize(x_scene[:,i:i+1])
            x_scene_pa += delta_b
            x_scene_temp[:,i:i+1] = DeParameterizeHomog(x_scene_pa)
            
            #compute cost
            epsilon_i = (Dehomogenize(x1[:,i:i+1]) -  Dehomogenize(P@x_scene_temp[:,i:i+1])) 
            epsilon_i_p = (Dehomogenize(x2[:,i:i+1]) -  Dehomogenize(P_p_temp@x_scene_temp[:,i:i+1]))
            cost_temp += computeCost(epsilon_i,epsilon_i_p,cov1,cov2)     
        ########  
        if(cost_temp > cost):
            lam = 10*lam
            #print ('iter %03d Cost %.9f'%(k+1, cost_temp))
        else:
            cost = cost_temp
            P_p = P_p_temp
            x_scene = x_scene_temp
            lam = 0.1*lam
            print ('iter %03d Cost %.9f'%(k+1, cost))
    
    print('Denormalization')
    P_p = np.linalg.inv(T2)@P_p@U
    F = FundamentalMatrixFromP(P_p)
    
    return F

if __name__ == '__main__':
    P_p = projectionMatrix_p(F_DLT)
    x_scene = triangulation(xin1,xin2,F_DLT,P_p) #Compute 3D scene points from triangulation
    
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    # Run LM initialized by DLT estimate
    print ('Sparse LM')
    time_start=time.time()
    F_LM = LM(F_DLT, xin1, xin2, max_iters, lam, x_scene, P_p)
    time_total=time.time()-time_start
    print('took %f secs'%time_total)
    
    # display the resulting F_LM, scaled with its frobenius norm
    DisplayResults(F_LM, 'F_LM')