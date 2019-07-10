from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2


class Part1(object):

    def get_features(self, img):
        img = cv2.imread(".\\part1-images\\" + img, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=1000)
        # detect features
        keypoints = orb.detect(img, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)
        keypoints, descriptors = orb.compute(img, keypoints)
        return keypoints, descriptors

    def features_collection(self, allfile):
        key_value = {}
        for file in allfile:
            key_value[file] = self.get_features(file)
        return key_value

    def match(self,file, matched, sa, sb, ratio=0.8):
        m = []
        for i in range(sa.shape[0]):
            s1 = []
            s2 = []
            dr1 = 10000
            dr2 = 10000
            tempj = 0
            for j in range(sb.shape[0]):
                d = cv2.norm(sa[i], sb[j], cv2.NORM_HAMMING)
                if d < dr1:
                    s2 = s1
                    dr2 = dr1
                    s1 = sb[j]
                    tempj = j
                    dr1 = d
                elif d < dr2:
                    s2 = sb[j]
                    dr2 = d
            if s2.size != 0 or dr1 / dr2 <= ratio:
                # m[matched] = sa[i], s1, dr1
                m.append([file, matched, i, tempj, dr1])
        # sort by distances
        m = np.asarray(m)
        m = sorted(m, key=lambda x: x[4])
        return m

    def k_means(self, data, k):
        n = data.shape[0]
        allDistances = data[:, 2]
        c = np.random.choice(range(n), k)
        for i in range(5):
            print(c)
            temp = allDistances
            c = np.array([])
            groups = np.argmin(temp, axis=0)
            for m in np.unique(groups):
                idx = np.where(groups == m)[0]
                temp2 = allDistances[idx, :]
                c = np.append(c, idx[np.argmin(np.sum(temp2, axis=1))])
            c = c.astype('int')
        return groups

    def draw_matches(self, imageA, imageB, kpsA, kpsB, matches):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for trainIdx, queryIdx in matches:

            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    def run(self, k, in_file, out_file):
        matched = []
        avg_distance = []
        # feature collection
        collection = self.features_collection(in_file)

        for file in in_file:
            for fc in collection:
                if fc != file:
                    mat = self.match(file, fc, collection[file][1], collection[fc][1])
                    avg_distance.append([file, fc, np.asarray(mat)[:, 4].astype(float).mean()])
                    matched.append(mat)
        # draw matched first image
        # collection['bigben_2.jpg'][0][keypoint index].pt gives x and y of keypoint

        # get Average distance for matched image pair
        # find clusters using image pair and distance info
        self.kmeans(avg_distance, k)

        # write to text file


if __name__ == "__main__":
    i = sys.argv.__len__()
    part = sys.argv[1]
    k = sys.argv[2]
    in_file = sys.argv[3:i-1]
    out_file = sys.argv[i-1]

    if part == 'part1':
        Part1().run(k, in_file,out_file)


    
    

