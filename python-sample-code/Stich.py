import numpy as np
import cv2
from scipy import spatial
import random

def getTranformationMatrix(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError
    A = np.zeros((1,8))
    for i in range(len(fp[0])):
        A = np.vstack((A, [fp[0,i], fp[1,i], 1, 0, 
                           0, 0, -fp[0,i]*tp[0,i], -fp[1,i]*tp[0,i]]))
        A = np.vstack((A, [0, 0, 0, fp[0,i], 
                           fp[1,i], 1, -fp[0,i]*tp[1,i], -fp[1,i]*tp[1,i]]))
    A = A[1:]
    B = (tp[:2].T).flatten().T
    A_inv = np.linalg.pinv(A)
    h = np.dot(A_inv, B)
    h = np.append(h,1).reshape(3,3)
    return h

def generateWarpedImage(img, k):
    warped = np.zeros(shape = (img.shape[0]*2,img.shape[1]*2,3)).astype('int')
    h = len(img)
    w = len(img[0])
    kInv = np.linalg.inv(k)

    for x in range(2*w):
        for y in range(2*h):
            co = np.dot(kInv, np.matrix([[x-w], [y-h], [1]]))
            xi, yi = co[0, 0]/co[2, 0], co[1, 0]/co[2, 0]
            if (xi < 0 or int(xi) >= w) or (yi < 0 or int(yi) >= h):
                pixel = [0, 0, 0]
            elif xi == int(xi) and yi == int(yi):
                pixel = img[int(yi), int(xi)]
            elif int(xi) < w and int(yi) < h:
                a = (yi - int(yi))
                b = (xi - int(xi))
                pixel = ((1 - b) * (1 - a) * img[int(yi), int(xi)])
                if int(yi) + 1 < h:
                    pixel += ((1 - b) * a * img[int(yi) + 1, int(xi)])
                if int(xi) + 1 < w:
                    pixel += (b * (1 - a) * img[int(yi), int(xi) + 1])
                if int(yi) + 1 < h and int(xi) + 1 < w:
                    pixel += (b * a * img[int(yi) + 1, int(xi) + 1])
            pixel = np.array(pixel)
            pixel[pixel > 255] = 255
            warped[y, x] = pixel
    return warped

def getMatches(img1, img2, T = 0.7):
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)
    alg = cv2.ORB_create()
    kps1, des1 = alg.detectAndCompute(im1, None)
    kps2, des2 = alg.detectAndCompute(im2, None)
    
    matchPairs = []
    for i in range(len(kps1)):
        dis = []
        for j in range(len(kps2)):
            dis.append(cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING))
        j1, j2 = np.argsort(dis)[:2]
        if dis[j1]/dis[j2] < T:
            matchPairs.append([kps1[i],kps2[j1]])
    return matchPairs

def extract_features(image_path):
    image = cv2.imread(image_path)
    alg = cv2.ORB_create()
    kps = alg.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)
    kps, dsc = alg.compute(image, kps)
    return kps, dsc

def getDistances(ds1, ds2):
    return spatial.distance.cdist(ds1, ds2, 'hamming')

def ransac(matchPairs, iterations = 1000, errorT = 25):
    inlierT = len(matchPairs)//2
    maxInliers = 0
    maxError = 0
    bestModel = None
    for i in range(iterations):
        random_patch = random.sample(matchPairs, 4)
        fp0, fp1, fp2 = [], [], []
        tp0, tp1, tp2 = [], [], []
        for corner in random_patch:
            fp0.append(corner[0].pt[0])
            fp1.append(corner[0].pt[1])
            fp2.append(1)
            tp0.append(corner[1].pt[0])
            tp1.append(corner[1].pt[1])
            tp2.append(1)
        fp = np.array([fp0, fp1, fp2])
        tp = np.array([tp0, tp1, tp2])
        H = getTranformationMatrix(fp, tp)
        H_votes = 0
        for p in matchPairs:
            x1, y1 = p[0].pt[0], p[0].pt[1]
            x2, y2 = p[1].pt[0], p[1].pt[1]
            A = np.array([x1, y1, 1]).reshape(3,1)
            B = np.array([x2, y2, 1]).reshape(3,1)
            B1 = np.dot(H, A)
            error = np.linalg.norm(B - B1) 
            if error < errorT:
                H_votes += 1
                if error > maxError: maxError = error
        if H_votes > inlierT and H_votes > maxInliers:
            maxInliers = H_votes
            bestModel = H
    return bestModel, maxInliers, maxError


def getTranslationForTargetImage(warpedSource, target):
    return 0, 0
    match = getMatches(warpedSource, target)

    im1pt1 = [match[0][0].pt[1], match[0][0].pt[0]]
    im1pt2 = [match[1][0].pt[1], match[1][0].pt[0]]
    im1pt3 = [match[2][0].pt[1], match[2][0].pt[0]]
    im2pt1 = [match[0][1].pt[1], match[0][1].pt[0]]
    im2pt2 = [match[1][1].pt[1], match[1][1].pt[0]]
    im2pt3 = [match[2][1].pt[1], match[2][1].pt[0]]
    
    sourcePoints = np.array([[im1pt1[0], im1pt2[0], im1pt3[0]], 
                             [im1pt1[1], im1pt2[1], im1pt3[1]]])
    targetPoints = np.array([[im2pt1[0], im2pt2[0], im2pt3[0]], 
                             [im2pt1[1], im2pt2[1], im2pt3[1]]])
    translation = np.mean(sourcePoints - targetPoints, axis = 1).astype('int')
    return translation[0], translation[1]

def generateTranslatedTargetImage(img2, transX, transY):
    im2 = cv2.imread(img2)
    im2_new = np.zeros((3*im2.shape[0],3*im2.shape[1],3))

    for i in range(len(im2)):
        for j in range(len(im2[0])):
            im2_new[i+transX][j+transY] = im2[i][j]
    return im2_new

def stitch(warpedSource, transTarget, output):
    match = getMatches(warpedSource, transTarget)
    im1 = cv2.imread(warpedSource)
    im2 = cv2.imread(transTarget)
    im1pt = [match[0][0].pt[1],match[0][0].pt[0]] 
    im2pt = [match[0][1].pt[1],match[0][1].pt[0]]
    
    length = im1pt[1] + (im2.shape[1] - im2pt[1])
    height = im1pt[0] + (im2.shape[0] - im2pt[0])
    
    im = np.zeros((int(height), int(length), 3))
    
    for i in range(len(im)):
        for j in range(len(im[0])):
            try:
                if all(im1[i,j] == [0,0,0]) and all(im2[i,j] != [0,0,0]):
                    im[i,j] = im2[i,j]
                elif all(im1[i,j] != [0,0,0]) and all(im2[i,j] == [0,0,0]):
                    im[i,j] = im1[i,j]
                elif all(im1[i,j] == [0,0,0]) and all(im2[i,j] == [0,0,0]):
                    im[i,j] = [0,0,0]
                else:
                    if any(im1[i,j] == [0,0,0]) or any(im2[i,j] == [0,0,0]):
                        temp = np.vstack((im1[i,j], im2[i,j]))
                        im[i,j] = np.max(temp, axis = 0)
                    else:
                        temp = np.vstack((im1[i,j], im2[i,j]))
                        im[i,j] = np.mean(temp, axis = 0).astype('int')
            except: pass
    cv2.imwrite(output, im)
    
    
basePath = 'F:/Study/2nd_Semester/CV/Assignments/a2/part2-images/'
img1 = 'temp1.jpg'
img2 = 'temp2.jpg'
output = 'blend.jpg'
warped_img1 = 'warpedSource.jpg'
trans_img2 = 'transTarget.jpg'

im1 = cv2.imread(basePath+img1)
im2 = cv2.imread(basePath+img2)

match = getMatches(basePath+img1, basePath+img2) 
h, _, _ = ransac(match)
warped = generateWarpedImage(im1, h)
cv2.imwrite(warped_img1, warped)
transX, transY = getTranslationForTargetImage(warped_img1, img2)
trans_target = generateTranslatedTargetImage(basePath+img2, transX, transY)
cv2.imwrite(trans_img2, trans_target)
stitch(warped_img1, trans_img2, output)
