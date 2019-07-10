import numpy as np
import cv2
import sys

class Part1:
    def clusterImages(self, k, imagePaths, output_file):
        N = len(imagePaths)
        allDistances = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i, N):
                allDistances[i, j] = self.getDistanceBetweenImages(imagePaths[i], imagePaths[j])
                allDistances[j, i] = allDistances[i, j]
        g = self.kMean(k, N, allDistances)

        classified = {}
        for i in np.argsort(g):
            if g[i] not in classified:
                classified[g[i]] = [imagePaths[i].split('/')[-1]]
            else:
                classified[g[i]].append(imagePaths[i].split('/')[-1])

        fo = open(output_file, "w")
        for grp in classified:
            lst = str(classified[grp])
            lst = lst.replace("', '", " ")
            lst = lst.replace("'", '')
            lst = lst.replace("[", '')
            lst = lst.replace("]", '')
            fo.writelines(lst + "\n")
        fo.close()

    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        alg = cv2.ORB_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)
        kps, dsc = alg.compute(image, kps)
        return kps, dsc

    def getMatches(self, img1, img2, T=0.7):
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
            if dis[j1] / dis[j2] < T:
                matchPairs.append([kps1[i], kps2[j1], dis[j1]])
        return matchPairs

    def getDistanceBetweenImages(self, p1, p2):
        if p1 == p2:
            return 0

        dis = 0
        match1 = self.getMatches(p1, p2)
        match2 = self.getMatches(p2, p1)
        for m in match1:
            dis += m[2]
        for m in match2:
            dis += m[2]
        if dis == 0:
            return sys.maxsize
        else:
            avg = dis/(len(match1) + len(match2))
            return avg

    def kMean(self, k, N, allDistances, itr = 10):
        c = np.random.choice(range(N), k)
        for i in range(itr):
            temp = allDistances[c,:]
            c = np.array([])
            groups = np.argmin(temp, axis=0)
            for m in np.unique(groups):
                idx = np.where(groups == m)[0]
                temp2 = allDistances[idx, :]
                c = np.append(c, idx[np.argmin(np.sum(temp2, axis=1))])
            c = c.astype('int')
        return groups