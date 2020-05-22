"""
Feature detection and Feature matching
@author: Szu-Hao Wu
"""

#%%Feature detection
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from scipy.ndimage.filters import gaussian_filter
from scipy import signal
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def ImageGradient(I):
    # inputs: 
    # I is the input image (may be mxn for Grayscale or mxnx3 for RGB)
    #
    # outputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    
    #m, n = I.shape[:2]
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = signal.convolve(I,kernelx)
    Iy = signal.convolve(I,kernely)
    
        
    return Ix, Iy
  

def MinorEigenvalueImage(Ix, Iy, w):
    # Calculate the minor eigenvalue image J
    #
    # inputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    # w is the size of the window used to compute the gradient matrix N
    #
    # outputs:
    # J0 is the mxn minor eigenvalue image of N before thresholding

    m, n = Ix.shape[:2]
    J0 = np.zeros((m,n))
    
    #Calculate your minor eigenvalue image J0.
    N = np.zeros((2,2,m,n))
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    
    index_shift = int((w-1)/2)
    for i in range(w,m-w):
        for j in range(w,n-w):
            N[0,0,i,j] = np.sum(Ixx[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            N[0,1,i,j] = np.sum(Ixy[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            N[1,0,i,j] = np.sum(Ixy[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            N[1,1,i,j] = np.sum(Iyy[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            
            J0[i,j] = (np.matrix.trace(N[:,:,i,j])) - np.sqrt(np.matrix.trace(N[:,:,i,j])**2 - 4*np.linalg.det(N[:,:,i,j]))/2 
            
    return J0,N
  
def NMS(J, w_nms):
    # Apply nonmaximum supression to J using window w_nms
    #
    # inputs: 
    # J is the minor eigenvalue image input image after thresholding
    # w_nms is the size of the local nonmaximum suppression window
    # 
    # outputs:
    # J2 is the mxn resulting image after applying nonmaximum suppression
    # 
    
    
    m = np.shape(J)[0]
    n = np.shape(J)[1]
    
    J2 = J.copy()
    J_temp = J.copy()
    m, n = J.shape[:2]
    index_shift = int((w_nms-1)/2)
    for i in range(m):
        for j in range(n):
            if(i<index_shift):
                i_min = 0
            else:
                i_min = i-index_shift
                
            if(i+index_shift>m):
                i_max = m
            else:
                i_max = i+index_shift
                
            if(j<index_shift):
                j_min = 0
            else:
                j_min = j-index_shift
                
            if(j+index_shift>n):
                j_max = n
            else:
                j_max = j+index_shift
                
            J_temp[i,j] = np.amax(J[i_min:i_max+1, j_min:j_max+1])
            
            if (J2[i,j] < J_temp[i,j]):
                J2[i,j] = 0
    
    
    return J2
  
  
def ForstnerCornerDetector(Ix, Iy, w, t, w_nms):
    # Calculate the minor eigenvalue image J
    # Threshold J
    # Run non-maxima suppression on the thresholded J
    # Gather the coordinates of the nonzero pixels in J 
    # Then compute the sub pixel location of each point using the Forstner operator
    #
    # inputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    # w is the size of the window used to compute the gradient matrix N
    # t is the minor eigenvalue threshold
    # w_nms is the size of the local nonmaximum suppression window
    #
    # outputs:
    # C is the number of corners detected in each image
    # pts is the 2xC array of coordinates of subpixel accurate corners
    #     found using the Forstner corner detector
    # J0 is the mxn minor eigenvalue image of N before thresholding
    # J1 is the mxn minor eigenvalue image of N after thresholding
    # J2 is the mxn minor eigenvalue image of N after thresholding and NMS

    m, n = Ix.shape[:2]
    J0 = np.zeros((m,n))
    J1 = np.zeros((m,n))

    #Calculate your minor eigenvalue image J0 and its thresholded version J1.
    J0,N = MinorEigenvalueImage(Ix, Iy, w)
    
    
    J1 = np.where(J0>t, J0, 0)
                
    #Run non-maxima suppression on your thresholded minor eigenvalue image.
    J2 = NMS(J1, w_nms)
    
    #Detect corners.
    C = 0
    b1 = np.zeros((m,n))
    b2 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            b1[i,j] = j*(Ix[i,j]**2)+i*Ix[i,j]*Iy[i,j]
            b2[i,j] = j*Ix[i,j]*Iy[i,j]+i*(Iy[i,j]**2)
    
    index_shift = int((w-1)/2)       
    A = N.copy()
    b = np.zeros((2,1,m,n))
    points = []
    for i in range(w,m-w):
        for j in range(w,n-w):
            b[0,0,i,j] = np.sum(b1[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            b[1,0,i,j] = np.sum(b2[i-index_shift:i+index_shift+1, j-index_shift:j+index_shift+1]) / (w*w)
            
            if(J2[i,j]>0):
                C += 1
                point = np.linalg.inv(A[:,:,i,j]).dot(b[:,:,i,j])
                points.append(point)    
    
    pts = np.zeros((2,C))
    for i in range(C):
        pts[0,i] = points[i][0]
        pts[1,i] = points[i][1]
    
    return C, pts, J0, J1, J2


# feature detection
def RunFeatureDetection(I, w, t, w_nms):
    Ix, Iy = ImageGradient(I)
    C, pts, J0, J1, J2 = ForstnerCornerDetector(Ix, Iy, w, t, w_nms)
    return C, pts, J0, J1, J2
    

if __name__ == '__main__':
    # input images
    I1 = np.array(Image.open('price_center20.JPG'), dtype='float')/255.
    I2 = np.array(Image.open('price_center21.JPG'), dtype='float')/255.
    I1_gray = rgb2gray(I1)
    I2_gray = rgb2gray(I2)
    # parameters to tune
    w = 9
    t = 0.0005
    w_nms = 9
    
    tic = time.time()
    # run feature detection algorithm on input images
    C1, pts1, J1_0, J1_1, J1_2 = RunFeatureDetection(I1_gray, w, t, w_nms)
    C2, pts2, J2_0, J2_1, J2_2 = RunFeatureDetection(I2_gray, w, t, w_nms)
    toc = time.time() - tic
    
    print('took %f secs'%toc)
    
    # display results
    plt.figure(figsize=(14,24))
    
    # show pre-thresholded minor eigenvalue images
    plt.subplot(3,2,1)
    plt.imshow(J1_0, cmap='gray')
    plt.title('pre-thresholded minor eigenvalue image')
    plt.subplot(3,2,2)
    plt.imshow(J2_0, cmap='gray')
    plt.title('pre-thresholded minor eigenvalue image')
    
    # show thresholded minor eigenvalue images
    plt.subplot(3,2,3)
    plt.imshow(J1_1, cmap='gray')
    plt.title('thresholded minor eigenvalue image')
    plt.subplot(3,2,4)
    plt.imshow(J2_1, cmap='gray')
    plt.title('thresholded minor eigenvalue image')
    
    # show corners on original images
    ax = plt.subplot(3,2,5)
    plt.imshow(I1)
    for i in range(C1): # draw rectangles of size w around corners
        x,y = pts1[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    plt.title('found %d corners'%C1)
    
    ax = plt.subplot(3,2,6)
    plt.imshow(I2)
    for i in range(C2):
        x,y = pts2[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    plt.title('found %d corners'%C2)
    
    plt.show()

#%%Feature matching
    
def NCC(I1, I2, pts1, pts2, w, p):
    # compute the normalized cross correlation between image patches I1, I2
    # result should be in the range [-1,1]
    #
    # inputs:
    # I1, I2 are the input images
    # pts1, pts2 are the point to be matched
    # w is the size of the matching window to compute correlation coefficients
    #
    # output:
    # normalized cross correlation matrix of scores between all windows in 
    #    image 1 and all windows in image 2
    
    """your code here"""
    I1 = rgb2gray(I1)
    I2 = rgb2gray(I2)
    m1,n1 = I1.shape[0],I1.shape[1]
    m2,n2 = I2.shape[0],I2.shape[1]
    s = int((w-1)/2)
    c1 = pts1.shape[1]
    c2 = pts2.shape[1]
    scores = np.zeros((c1,c2))
    
    for i in range(c1):
        for j in range(c2):
            y_1 = int(pts1[1,i])
            x_1 = int(pts1[0,i])
            y_2 = int(pts2[1,j])
            x_2 = int(pts2[0,j])
        
            i_start1 = y_1-s
            i_end1 = y_1+s+1 
            j_start1 = x_1-s
            j_end1 = x_1+s+1
            i_start2 = y_2-s
            i_end2 = y_2+s+1
            j_start2 = x_2-s
            j_end2 = x_2+s+1
            
            if (i_start1 < 0 or i_end1 > m1 or j_start1 < 0 or j_end1 >
                n1 or i_start2 < 0 or i_end2 > m2 or j_start2 < 0 or j_end2 > n2):
                continue
         
            
            W1 = I1[i_start1:i_end1,j_start1:j_end1]
            W2 = I2[i_start2:i_end2,j_start2:j_end2]
            W1_nor = (W1-np.mean(W1))/np.sqrt(np.sum((W1-np.mean(W1))**2))
            W2_nor = (W2-np.mean(W2))/np.sqrt(np.sum((W2-np.mean(W2))**2))
            if (np.sqrt((y_1-y_2)**2 + (x_1-x_2)**2) > p):
                continue 
            scores[i,j] = np.sum(W1_nor*W2_nor)
    
        
    
    return scores


def Match(scores, t, d, p):
    # perform the one-to-one correspondence matching on the correlation coefficient matrix
    # 
    # inputs:
    # scores is the NCC matrix
    # t is the correlation coefficient threshold
    # d distance ration threshold
    # p is the size of the proximity window (check whether the point is too far away)
    #
    # output:
    # list of the feature coordinates in image 1 and image 2 
    
    """your code here"""
    inds = []
    m,n = scores.shape[0],scores.shape[1]
    mask = np.ones((m,n))
    
    while np.amax(scores)>t:
        scores = mask*scores
        max_value = np.amax(scores)
        max_index = np.where(scores == max_value)
        mask[max_index[0][0],max_index[1][0]] = 0
        
        temp = mask*scores
        second_value_row = np.amax(temp[max_index[0][0],:])
        second_value_col = np.amax(temp[:,max_index[1][0]])
        if second_value_row > second_value_col: second_value = second_value_row
        else: second_value = second_value_col
        
        
        if (1-max_value)<(1-second_value)*d:
            inds.append([max_index[0][0],max_index[1][0]])
            mask[max_index[0][0],:] = 0
            mask[:,max_index[1][0]] = 0
    inds = np.array(inds).T
    
    return inds


def RunFeatureMatching(I1, I2, pts1, pts2, w, t, d, p):
    # inputs:
    # I1, I2 are the input images
    # pts1, pts2 are the point to be matched
    # w is the size of the matching window to compute correlation coefficients
    # t is the correlation coefficient threshold
    # d distance ration threshold
    # p is the size of the proximity window
    #
    # outputs:
    # inds is a 2xk matrix of matches where inds[0,i] indexs a point pts1 
    #     and inds[1,i] indexs a point in pts2, where k is the number of matches
    
    scores = NCC(I1, I2, pts1, pts2, w, p)
    inds = Match(scores, t, d, p)
    return inds,scores

if __name__ == '__main__':
    # parameters to tune
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
