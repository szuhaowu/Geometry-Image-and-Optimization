"""
Feture Matching
@author: Szu-Hao Wu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from Feature_Detection import rgb2gray

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