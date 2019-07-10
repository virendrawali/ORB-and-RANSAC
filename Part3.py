import numpy as np
import cv2
import random

class Part3:
    
    def __init__(self):
        self.warped_img1 = 'warpedSource.jpg'
        self.trans_img2 = 'transTarget.jpg'
      
    def getTranformationMatrix(self, fp, tp):
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
    
    def generateWarpedImage(self, img, k):
        h = len(img)
        w = len(img[0])
        kInv = np.linalg.pinv(k)
        corners = [[0,0],[0,w],[h,0],[h,w]]
        corner_x, corner_y, corner_w = [], [], []
        
        for corner in corners:
            corner_x.append(corner[0])
            corner_y.append(corner[1])
            corner_w.append(1)
        
        corner_matrix = np.array([corner_x, corner_y, corner_w])
        warped_corner_matrix = np.dot(k,corner_matrix)
        max_X, max_Y = np.max(warped_corner_matrix, axis = 1)[:2]
        min_X, min_Y = np.min(warped_corner_matrix, axis = 1)[:2]
        if(min_X > 0): min_X = 0 
        if(min_Y > 0): min_Y = 0
        
        warped = np.zeros(shape = (int(2*max_Y-min_Y)*2,int(2*max_X-min_X)*2,3))
        warped = warped.astype('int')
        
        # Bi-linear interpolation
        for x in range(int(min_X-1),int(max_X+1)*2):
            for y in range(int(min_Y-1),int(max_Y+1)*2):
                pointInSourceImage = np.dot(kInv, np.matrix([[x], [y], [1]]))
                xi = pointInSourceImage[0, 0]/pointInSourceImage[2, 0] 
                yi = pointInSourceImage[1, 0]/pointInSourceImage[2, 0]
                
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
                if min_X<0 or min_Y<0:
                    warped[y-int(min_Y), x - int(min_X)] = pixel
                else:
                    warped[y, x] = pixel
        return warped
    
    def getMatches(self, img1, img2, T = 0.7):
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
            d = dis[j1]/dis[j2]
            if d < T:
                matchPairs.append([kps1[i],kps2[j1],d])
        
        matchPairs.sort(key=lambda tup: tup[2])
        return matchPairs
    
    def ransac(self, matchPairs, iterations = 1000, errorT = 25):
        # Model should have atleast half points as inliers
        maxInliers = 0
        bestModel = None
        
        for i in range(iterations):
            #Generating model (transformation matrix) from random 4 points
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
            H = self.getTranformationMatrix(fp, tp)
            
            # Voting (counting # of inliers) for the model
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
            
            # Getting model having max # of inliers        
            if H_votes > maxInliers:
                maxInliers = H_votes
                bestModel = H
        return bestModel
    
    def getTranslationForTargetImage(self, warpedSource, target):
        # Getting 3 match points to calculate required translation
        match = self.getMatches(warpedSource, target)
        
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
    
    def generateTranslatedTargetImage(self, img2, transX, transY):
        # Generating translated image
        im2 = cv2.imread(img2)
        im2_new = np.zeros((3*im2.shape[0],3*im2.shape[1],3))
    
        for i in range(len(im2)):
            for j in range(len(im2[0])):
                im2_new[i+transX][j+transY] = im2[i][j]
        return im2_new
    
    def stitch(self, warpedSource, transTarget, output):
        # Stitching two images based on 1 match pair in both images 
        match = self.getMatches(warpedSource, transTarget)
            
        im1 = cv2.imread(warpedSource)
        im2 = cv2.imread(transTarget)
        im1pt = [match[0][0].pt[1],match[0][0].pt[0]] 
        im2pt = [match[0][1].pt[1],match[0][1].pt[0]]
        
        length = im1pt[1] + (im2.shape[1] - im2pt[1])
        height = im1pt[0] + (im2.shape[0] - im2pt[0])
        
        # Generating final stiched image
        im = np.zeros((int(height), int(length), 3))
        for i in range(len(im)):
            for j in range(len(im[0])):
                try:
                    # If pixal co-ordinate not present in one of the image then copy pixal value of another image.
                    if (i >= len(im1) or j >= len(im1[0])) and (i >= len(im2) or j >= len(im2[0])): 
                        im[i,j] = [0,0,0]
                    elif (i >= len(im1) or j >= len(im1[0])) and not (i >= len(im2) or j >= len(im2[0])): 
                        im[i,j] = im2[i,j]
                    elif (i >= len(im2) or j >= len(im2[0])) and not (i >= len(im1) or j >= len(im1[0])): 
                        im[i,j] = im1[i,j]
                    # If pixal value of one of the image is 0, then copy pixal value of another image.
                    elif all(im1[i,j] == [0,0,0]) and all(im2[i,j] != [0,0,0]):
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
                            #Otherwise, put avearage of pixal values of both images.
                            temp = np.vstack((im1[i,j], im2[i,j]))
                            im[i,j] = np.mean(temp, axis = 0).astype('int')
                except: pass
        cv2.imwrite(output, im)