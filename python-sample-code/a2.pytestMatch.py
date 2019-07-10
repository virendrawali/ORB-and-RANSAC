# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:20:46 2019

@author: Darshan
"""

import numpy as np
import cv2
from scipy import spatial
from random import choice
from scipy import ndimage
from PIL import Image

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    alg = cv2.ORB_create()
    kps = alg.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)
    kps, dsc = alg.compute(image, kps)

    for i in range(len(kps)):
        for j in range(-5,5):
            image[int(kps[i].pt[1])+j, int(kps[i].pt[0])+j] = 0 
            image[int(kps[i].pt[1])-j, int(kps[i].pt[0])+j] = 255
    cv2.imwrite(image_path.split('/')[-1], image)
    return kps, dsc

def getDistances(ds1, ds2):
    return spatial.distance.cdist(ds1, ds2, 'hamming')

def getMatches(p1, p2):
    kp1, ds1 = extract_features(p1)
    kp2, ds2 = extract_features(p2)
    
    d = getDistances(ds1,ds2)
    m = np.argsort(d,axis = 1)
    n = []
    i = 0
    for row in m:
        ratio = d[i,row[0]] / d[i,row[1]]
        idx = row[0]
        n.append([ratio, idx])
        i += 1
    n = np.array(n)
    match = []
    for j in range(len(n)):
        if n[j,0] < 0.77:
            start = ds1[j].reshape(1,-1)
            end = ds2[int(n[j,1])].reshape(1,-1)
            d = getDistances(start, end)[0,0]
            match.append((kp1[j], kp2[int(n[j,1])], d))
    
    match.sort(key=lambda tup: tup[2])
    return match

def getDistance(p1, p2):
    match1 = getMatches(p1, p2)
    match2 = getMatches(p2, p1)
    
    dis = 0
    for m in match1:
        dis += m[2]
    for m in match2:
        dis += m[2]
    if dis == 0: 
        return 'No match found'
    else:
        avg = dis/(len(match1) + len(match2))
        return avg 

def appendimages(im1, im2):
    """ Return a new concatenated images side-by-side """
    if np.ndim(im1) == 2:
        return _appendimages(im1, im2)
    else:
        imr = _appendimages(im1[:, :, 0], im2[:, :, 0])
        img = _appendimages(im1[:, :, 1], im2[:, :, 1])
        imb = _appendimages(im1[:, :, 2], im2[:, :, 2])
        return np.dstack((imr, img, imb))


def _appendimages(im1,im2):
    """ return a new image that appends the two images side-by-side."""

    #select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))), axis=0)
    else:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))), axis=0)

    return np.concatenate((im1,im2), axis=1)

def ransac(im1, im2, points_list, iters = 10, error = 10, good_model_num = 4):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''

    if np.ndim(im1) == 2:
        rows,cols = im1.shape
    else:
        rows, cols, _ = im1.shape

    model_error = 511
    model_H = None

    for i in range(iters):
        # Randomly select 3 points
        temp_point_list = np.copy(points_list).tolist()
        consensus_set = randomPoints(temp_point_list, 4)

        # Calculate the homography matrix from the 3 points

        fp0 = []
        fp1 = []
        fp2 = []

        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:
            # X and Y co-ordinates are inverse just for testing... If not worked then dont forget to revert it back
            # Just make pt[0] for fp0, tp0 and pt[1] for fp1, tp1 
            fp0.append(line[0].pt[1])
            fp1.append(line[0].pt[0])
            fp2.append(1)

            tp0.append(line[1].pt[1])
            tp1.append(line[1].pt[0])
            tp2.append(1)

        fp = np.array([fp0, fp1, fp2])
        tp = np.array([tp0, tp1, tp2])

        H = Haffine_from_points(fp, tp)

        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in temp_point_list:
            x1, y1 = p[0].pt[0], p[0].pt[1]
            x2, y2 = p[1].pt[0], p[1].pt[1]

            A = np.array([x1, y1, 1]).reshape(3,1)
            B = np.array([x2, y2, 1]).reshape(3,1)

            out = B - np.dot(H, A)
            dist_err = np.hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)


        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0].pt[0], p[0].pt[1]
                x1, y1 = p[1].pt[0], p[1].pt[1]

                A = np.array([x0, y0, 1]).reshape(3,1)
                B = np.array([x1, y1, 1]).reshape(3,1)

                out = B - np.dot(H, A)
                dist_err = np.hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if ((max(dists) < model_error)):#and max(dists) < error)):
                model_error = max(dists)
                model_H = H

    return model_H

def randomPoints(points, k):
    out = []
    for j in range(k):
        temp = choice(points)
        out.append(temp)
        points.remove(temp)
    return out

def Haffine_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError
    A = np.zeros((1,8))
    
    for i in range(len(fp[0])):
        A = np.vstack((A, [fp[0,i], fp[1,i], 1, 0, 0, 0, -fp[0,i]*tp[0,i], -fp[1,i]*tp[0,i]]))
        A = np.vstack((A, [0, 0, 0, fp[0,i], fp[1,i], 1, -fp[0,i]*tp[1,i], -fp[1,i]*tp[1,i]]))
    A = A[1:]
    
    if np.linalg.det(A) != 0:
        A_inv = np.linalg.inv(A)
        B = (tp[:2].T).flatten().T
        h = np.dot(A_inv, B)
        return np.append(h,1).reshape(3,3)
    else:
        idx = randomPoints(list(range(4)), 3)
        fp1 = fp[:,idx]
        tp1 = tp[:,idx]
        A = np.zeros((1,6))
        
        for i in range(len(fp1[0])):
            A = np.vstack((A,[fp1[0,i], fp1[1,i], 1, 0, 0, 0]))
            A = np.vstack((A,[0, 0, 0, fp1[0,i], fp1[1,i], 1]))
        A = A[1:]
        
        if np.linalg.det(A) != 0:
            A_inv = np.linalg.inv(A)
            B = (tp1[:2].T).flatten().T
            h = np.dot(A_inv, B)
            return np.append(h,[0,0,1]).reshape(3,3)
        else:
            idx = randomPoints(list(range(4)), 2)
            fp2 = fp[:,idx]
            tp2 = tp[:,idx]
            A = np.zeros((1,4))
            
            for i in range(len(fp2[0])):
                A = np.vstack((A,[fp2[0,i], fp2[1,i], 0, 0]))
                A = np.vstack((A,[0, 0, fp2[0,i], fp2[1,i]]))
            A = A[1:]
            A_inv = np.linalg.inv(A)
            B = (tp2[:2].T).flatten().T
            h = np.dot(A_inv, B)
            h = np.insert(h,2,0)
            h = np.insert(h,5,0)
            return np.append(h,[0,0,1]).reshape(3,3)
        
def affine_transform2(im, rot, shift):
    '''
        Perform affine transform for 2/3D images.
    '''
    if np.ndim(im) == 2:
        return ndimage.affine_transform(im, rot, shift)
    else:
        imr = ndimage.affine_transform(im[:, :, 0], rot, shift)
        img = ndimage.affine_transform(im[:, :, 1], rot, shift)
        imb = ndimage.affine_transform(im[:, :, 2], rot, shift)

        return np.dstack((imr, img, imb))

basePath = 'F:/Study/2nd_Semester/CV/Assignments/a2/part2-images/'
p1 = 'book1.jpg'
p2 = 'book2.jpg'
im1 = cv2.imread(basePath + p1, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(basePath + p2, cv2.IMREAD_GRAYSCALE)
im = appendimages(im1,im2)

l = len(im1[0])

match = getMatches(basePath+p1, basePath+p2)

'''
for m in match[:50]:
    cv2.line(im, (int(m[0].pt[0]), int(m[0].pt[1])), (l + int(m[1].pt[0]), int(m[1].pt[1])), 255, 1)
    
cv2.imwrite('temp.jpg', im)
'''
im1 = np.asarray(Image.open(basePath + p1).convert('L'))
im2 = np.asarray(Image.open(basePath + p2).convert('L'))

out_ransac = ransac(im1, im2, match)
H_ransac = np.linalg.inv(out_ransac)
im_ransac = affine_transform2(im1,
                                  H_ransac[:2, :2],
                                  [H_ransac[0][2], H_ransac[1][2]])
Image.fromarray(im_ransac).show()
