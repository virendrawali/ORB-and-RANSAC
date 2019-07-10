import numpy as np

class Part2:
    def getTransformationMatrix(self, n, points):
        
        # Generate matrix of points in original image as fp and points in transformed image as tp
        from_pt = []
        to_pt = []
        for i in range(0,len(points), 4):
            from_pt.append((points[i], points[i+1]))
            to_pt.append((points[i+2], points[i+3]))
        fp0, fp1, fp2 = [], [], []
        tp0, tp1, tp2 = [], [], []
        
        for point in from_pt:
            fp0.append(point[0])
            fp1.append(point[1])
            fp2.append(1)
            
        for point in to_pt:
            tp0.append(point[0])
            tp1.append(point[1])
            tp2.append(1)
            
        fp = np.array([fp0, fp1, fp2])
        tp = np.array([tp0, tp1, tp2])
        
        # Generate transformation matrix for original and transformed points based on transformation type
        if n == 1: # Translation
            transX = fp[0,0] - tp[0,0] 
            transY = fp[1,0] - tp[1,0]
            return np.array([[1,0,transX],[0,1,transY],[0,0,1]])
        
        if n == 2: # Euclidean (rigid) transformation
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
        
        if n == 3: # Affine transformation
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
        
        if n == 4: # Projective transformation
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
        
    def generateWarpedImage(self, img, k):
        
        #Generate warped image of original image using transformation matrix
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