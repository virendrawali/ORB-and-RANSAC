import numpy as np
import cv2

def getTransformationMatrix(n,list1):
    from_pt = []
    to_pt = []
    for i in range(0,len(list1),4):
        from_pt.append((list1[i],list1[i+1]))
        to_pt.append((list1[i+2],list1[i+3]))
    fp0, fp1, fp2 = [], [], []
    tp0, tp1, tp2 = [], [], []
    
    for line in from_pt:
        fp0.append(line[0])
        fp1.append(line[1])
        fp2.append(1)
    for line in to_pt:
        tp0.append(line[0])
        tp1.append(line[1])
        tp2.append(1)
    fp = np.array([fp0, fp1, fp2])
    tp = np.array([tp0, tp1, tp2])
    
    print(fp)
    print(tp)
    
    if n == 4:
        A = np.zeros((1,8))
        for i in range(len(fp[0])):
            A = np.vstack((A, [fp[0,i], fp[1,i], 1, 0, 
                               0, 0, -fp[0,i]*tp[0,i], -fp[1,i]*tp[0,i]]))
            A = np.vstack((A, [0, 0, 0, fp[0,i], 
                               fp[1,i], 1, -fp[0,i]*tp[1,i], -fp[1,i]*tp[1,i]]))
        A = A[1:]
        if np.linalg.det(A) != 0:
            A_inv = np.linalg.inv(A)
            B = (tp[:2].T).flatten().T
            h = np.dot(A_inv, B)
            return np.append(h,1).reshape(3,3)
    
    if n == 3:
        A = np.zeros((1,6))
        for i in range(len(fp[0])):
            A = np.vstack((A,[fp[0,i], fp[1,i], 1, 0, 0, 0]))
            A = np.vstack((A,[0, 0, 0, fp[0,i], fp[1,i], 1]))
        A = A[1:]
        if np.linalg.det(A) != 0:
            A_inv = np.linalg.inv(A)
            B = (tp[:2].T).flatten().T
            h = np.dot(A_inv, B)
            return np.append(h,[0,0,1]).reshape(3,3)
    
    if n == 2:
        A = np.zeros((1,4))
        for i in range(len(fp[0])):
            A = np.vstack((A,[fp[0,i], fp[1,i], 0, 0]))
            A = np.vstack((A,[0, 0, fp[0,i], fp[1,i]]))
        A = A[1:]
        A_inv = np.linalg.pinv(A)
        B = (tp[:2].T).flatten().T
        h = np.dot(A_inv, B)
        h = np.insert(h,2,0)
        h = np.insert(h,5,0)
        return np.append(h,[0,0,1]).reshape(3,3)
    
    if n == 1:
        transX = fp[0,0] - tp[0,0] 
        transY = fp[1,0] - tp[1,0]
        return np.array([[1,0,transX],[0,1,transY],[0,0,1]])

def generateWarpedImage(img, k):
    warped = np.zeros(shape = img.shape).astype('int')
    h = len(img)
    w = len(img[0])
    kInv = np.linalg.pinv(k)

    for x in range(w):
        for y in range(h):
            co = np.dot(kInv, np.matrix([[x], [y], [1]]))
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

n = 4
basePath = 'part2-images/'
p1 = 'book1.jpg'
p2 = 'book2.jpg'
im1 = cv2.imread(basePath + p1)
im2 = cv2.imread(basePath + p2)
#list1 = [318,256,141,131,534,372,480,159,316,670,493,630,73,473,64,601] # for book-1
list1 = [141,131,318,256,480,159,534,372,493,630,316,670,64,601,73,473]  # for book-2
kernel = getTransformationMatrix(n,list1)
print(kernel)
warped = generateWarpedImage(im2, kernel)
cv2.imwrite('war_temp.jpg',warped)
