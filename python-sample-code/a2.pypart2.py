import cv2
import numpy as np
from PIL import  Image

def img_warper(img, k):
    """
        Function that takes an input image and applies a given 3x3 coordinate transformation matrix
        (using homogeneous coordinates) to produce a corresponding warped image.
    :param img: input image
    :param k: kernel matrix
    :return: warped image
    """

    h, w, _ = img.shape
    warped = np.zeros(shape = img.shape)
    print(warped.shape)
    print(img.shape)

    # for Inverse WarpingWarping, will need to take inverse of Kernel matrix
    kInv = np.linalg.inv(k)

    for x in range(w):
        for y in range(h):
            co = np.dot(kInv, np.matrix([[x], [y], [1]]))
            xi, yi = co[0, 0]/co[2, 0], co[1, 0]/co[2, 0]

            # if xi and yi falls outsize the original image boundary then its unmapped region
            if (xi < 0 or int(xi) >= w) or (yi < 0 or int(yi) >= h):
                pixel = [0, 0, 0]

            # if it maps to the actual location in original image
            elif xi == int(xi) and yi == int(yi):
                pixel = img[int(yi), int(xi)]

            # Bilinear Interpolation
            elif int(xi) < w and int(yi) < h:
                # print(yi, xi)
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

    cv2.imwrite("lincoln-orb.jpg", warped)
    # Image.fromarray(warped.astype(np.uint8)).show()


img = cv2.imread("part2-images\\book1.jpg")

k = np.array([
    [    0.907, 0.258,   -182],
    [   -0.153, 1.44,      58],
    [-0.000306, 0.000731,   1]
  ])

img_warper(img, k)
