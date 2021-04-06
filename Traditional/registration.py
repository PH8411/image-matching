import cv2
import numpy as np

MIN_MATCH_COUNT = 10

def SIFT_REGIS(src_img,temp_img,RESIZE_SCALE=None,MATCH_VIZ=False):
    height,width = temp_img.shape[:2]

    if RESIZE_SCALE is not None:
        src_img=cv2.resize(src_img,(int(RESIZE_SCALE*width),int(RESIZE_SCALE*height)),interpolation=cv2.INTER_CUBIC)#source image
        temp_img=cv2.resize(temp_img,(int(RESIZE_SCALE*width),int(RESIZE_SCALE*height)),interpolation=cv2.INTER_CUBIC)#template image

    image1 = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)

    siftDetector = cv2.xfeatures2d.SIFT_create()
    keyPoint1, imageDesc1 = siftDetector.detectAndCompute(image1, None)
    keyPoint2, imageDesc2 = siftDetector.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matchePoints = flann.knnMatch(imageDesc1, imageDesc2, k=2) 

    good = []
    for m,n in matchePoints:
        if m.distance < 0.7*n.distance:
            good.append(m)  

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([keyPoint1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([keyPoint2[m.trainIdx].pt for m in good])
        # M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=7)   
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=7)  
        if MATCH_VIZ:
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 0)
            match_img = cv2.drawMatches(src_img,keyPoint1,temp_img,keyPoint2,good,None,**draw_params)  
        else:
            match_img=None              
    else:
        print ("SIFT:Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        return None
   
    return M, match_img

def ORB_REGIS(src_img,temp_img,RESIZE_SCALE=None,MATCH_VIZ=False):
    height,width = temp_img.shape[:2]

    if RESIZE_SCALE is not None:
        src_img=cv2.resize(src_img,(int(RESIZE_SCALE*width),int(RESIZE_SCALE*height)),interpolation=cv2.INTER_CUBIC)#source image
        temp_img=cv2.resize(temp_img,(int(RESIZE_SCALE*width),int(RESIZE_SCALE*height)),interpolation=cv2.INTER_CUBIC)#template image

    image1 = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
    
    ORB=cv2.ORB_create()
    keyPoint1, imageDesc1 = ORB.detectAndCompute(image1, None)
    keyPoint2, imageDesc2 = ORB.detectAndCompute(image2, None)

    #暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(imageDesc1,imageDesc2)
    matches = sorted(matches, key = lambda x:x.distance)

    if len(matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([keyPoint1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keyPoint2[m.trainIdx].pt for m in matches])
        # M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=7)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=7)
        if MATCH_VIZ:
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 0)
            match_img = cv2.drawMatches(src_img,keyPoint1,temp_img,keyPoint2,matches,None,**draw_params)
        else:
            match_img=None                
    else:
        print ("ORB:Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        return None
   
    return M,match_img

