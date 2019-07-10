import numpy as np
import cv2
from scipy import spatial
import sys
import os
import os.path

basePath = 'F:/Study/2nd_Semester/CV/Assignments/a2/part1-images/'
imagePaths = [os.path.join(basePath, p) for p in sorted(os.listdir(basePath))][:20]
N = len(imagePaths)

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
    #cv2.imwrite(image_path.split('/')[-1], image)
    return kps, dsc

def getDistances(ds1, ds2):
    return spatial.distance.cdist(ds1, ds2, 'cosine')

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
        if n[j,0] > 0.8:
            start = ds1[j].reshape(1,-1)
            end = ds2[int(n[j,1])].reshape(1,-1)
            d = getDistances(start, end)[0,0]
            match.append((kp1[j], kp2[int(n[j,1])], d))
    
    match.sort(key=lambda tup: tup[2])
    return match

def getDistance(p1, p2):
    if p1 == p2:
        return 0
    
    dis = 0
    match1 = getMatches(p1, p2)
    match2 = getMatches(p2, p1)
    
    for m in match1:
        dis += m[2]
    for m in match2:
        dis += m[2]
    if dis == 0: 
        return sys.maxsize
    else:
        avg = dis/(len(match1) + len(match2))
        return avg

allDistances = np.zeros((N,N))
if os.path.exists(basePath.split('/')[-2] + '.txt'):
    allDistances = np.loadtxt(basePath.split('/')[-2] + '.txt')
else:
    for i in range(N):
        for j in range(i, N):
            allDistances[i,j] = getDistance(imagePaths[i], imagePaths[j])
            allDistances[j,i] = allDistances[i,j]
    np.savetxt(basePath.split('/')[-2] + '.txt', allDistances)
        
def kMean(k):
    c = np.random.choice(range(N), k)
    for i in range(5):
        print(c)
        temp = allDistances[c,:]
        c = np.array([])
        groups = np.argmin(temp, axis = 0)
        for m in np.unique(groups):
            idx = np.where(groups == m)[0]
            temp2 = allDistances[idx,:]
            c = np.append(c,idx[np.argmin(np.sum(temp2, axis = 1))])
        c = c.astype('int')
    return groups

g = kMean(2)
            
        