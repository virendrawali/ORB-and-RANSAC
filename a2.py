#!/usr/local//bin/python3

import sys
import numpy as np
import cv2
from Part1 import Part1
from Part2 import Part2
from Part3 import Part3


if __name__ == "__main__":
    if sys.argv[1] == "part1":
        k = int(sys.argv[2])
        images = sys.argv[3:len(sys.argv)-1]
        output_file = sys.argv[-1]

        p1 = Part1()
        p1.clusterImages(k, images, output_file)

    if sys.argv[1] == "part2":
        n = int(sys.argv[2])
        input_image1 = sys.argv[3]
        input_image2 = sys.argv[4]
        
        p2 = Part2()
        im1 = cv2.imread(input_image1)
        im2 = cv2.imread(input_image2)
        output_image = sys.argv[5]
        points = np.array(sys.argv[6:]).astype('int')
        transformation_matrix = p2.getTransformationMatrix(n,points)
        warped = p2.generateWarpedImage(im2, transformation_matrix)
        cv2.imwrite(output_image, warped)
        
    if sys.argv[1] == "part3":
        input_image1 = sys.argv[2]
        input_image2 = sys.argv[3]
        im1 = cv2.imread(input_image1)
        im2 = cv2.imread(input_image2)
        output_image = sys.argv[4]
        
        p3 = Part3()
        match = p3.getMatches(input_image1, input_image2) 
        h = p3.ransac(match)
        print('Generating warped image...')
        warped = p3.generateWarpedImage(im1, h)
        cv2.imwrite(p3.warped_img1, warped)
        transX, transY = p3.getTranslationForTargetImage(p3.warped_img1, input_image2)
        trans_target = p3.generateTranslatedTargetImage(input_image2, transX, transY)
        print('Generating translated image...')
        cv2.imwrite(p3.trans_img2, trans_target)
        print('Generating final merged image...')
        p3.stitch(p3.warped_img1, p3.trans_img2, output_image)