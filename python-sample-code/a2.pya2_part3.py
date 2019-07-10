import numpy as np
import sys
import cv2

def get_features(img):
    img = cv2.imread(".\\part1-images\\" + img, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=1000)
    # detect features
    keypoints = orb.detect(img, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints, descriptors = orb.compute(img, keypoints)
    return keypoints, descriptors


def match(matched, sa, sb, ratio=0.8):
    m = {}
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
        if s2.size != 0 or dr1/dr2 <= ratio:
            # m[matched] = sa[i], s1, dr1
            m[i] = tempj, dr1

    #m = sorted(m.items(), key=lambda x: x[2])
    return m


'''
https://en.wikipedia.org/w/index.php?title=Random_sample_consensus
Given:
    data – a set of observations
    model – a model to explain observed data points
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data

Return:
    bestFit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestFit = nul
bestErr = something really large
while iterations < k {
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    for every point in data not in maybeInliers {
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    }
    if the number of elements in alsoInliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if thisErr < bestErr {
            bestFit = betterModel
            bestErr = thisErr
        }
    }
    increment iterations
}
return bestFit
'''


def ransac(data, model, n, k, t, d):
    iterations = 0
    bestfit = None
    besterr = np.inf
    while iterations < k:
        # n randomly selected values from data
        index = np.random.choice(data, n, replace=False)
        maybeInliers = data[index]

        # model parameters fitted to maybeInliers
        # some model not sure now
        maybeModel = model.fit(maybeInliers)

        alsoInliers = {}




        iterations+=1


if __name__ == "__main__":
    # 1 extract interest points from each image
    var = 0
    # 2 figure the relative transformation between the images by implementing RANSAC

    # 3 transform the images into a common coordinate system

    # 4 blend the images together (e.g. by simply averaging them pixel-by-pixel) to produce out-put.jpg





