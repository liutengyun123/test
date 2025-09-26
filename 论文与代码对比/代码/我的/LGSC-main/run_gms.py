import cv2
import numpy as np
import testmodel

from enum import Enum

def siftandgetlocation(img1,img2,goodflag=1):
    sift = cv2.SIFT_create(500)

    # find the keypoints and descriptors with SIFT
    loc1, des1 = sift.detectAndCompute(img1, None)
    loc2, des2 = sift.detectAndCompute(img2, None)
    # abb = cv2.drawKeypoints(img2, loc2, img2, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  #
    # cv2.imshow('abb', abb)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 1 * n.distance:
            good.append(m)
    print('num of matching points', len(good))

    return [good,loc1,loc2]


def run_gms(img1,img2,H):
    ori_img = cv2.imread(img1, 0)
    dst_img = cv2.imread(img2, 0)

    result = siftandgetlocation(ori_img, dst_img, 1)
    matches = result[0]
    ori_kp = result[1]
    dst_kp = result[2]
    H = np.array(H)


    print('the correct matches before removal:')
    before,count_before = testmodel.testscoreimage(matches, ori_kp, dst_kp, H, ori_img, dst_img)

    #使用gms剔除误匹配
    matches_gms = cv2.xfeatures2d.matchGMS(ori_img.shape[:2], ori_img.shape[:2], ori_kp, dst_kp, matches, withScale=False, withRotation=False,thresholdFactor=6)

    final_matches = []
    for m in matches_gms:
        final_matches.append(m)

    print(len(final_matches),len(matches_gms))
    #通过H矩阵计算正确的匹配点对
    after,count_after = testmodel.testscoreimage(final_matches, ori_kp, dst_kp, H,ori_img,dst_img)

    if count_before ==0 or len(final_matches) == 0:
        return 0,0,0

    recall = count_after/count_before
    precision = count_after/len(final_matches)
    F_score = 2*recall*precision/(recall+precision)
    
    cv2.imshow('after', after)
    cv2.waitKey()

    return recall,precision,F_score

if __name__ == '__main__':
    # img1 = 'E:\paper_testcode\match\\VGG\\bikes\\img1.ppm'
    # img2 = 'E:\paper_testcode\match\\VGG\\bikes\\img2.ppm'
    
    # img1 = 'E:\paper_testcode\match\\VGG\\ubc\\img1.ppm'
    # img2 = 'E:\paper_testcode\match\\VGG\\ubc\\img2.ppm'
    
    img1 = 'E:\paper_testcode\match\\VGG\\wall\\img1.ppm'
    img2 = 'E:\paper_testcode\match\\VGG\\wall\\img2.ppm'
     
    
    # Hlist1_2 = [[   1.0107879e+00,   8.2814684e-03,   1.8576800e+01],
    # [-4.9128885e-03 ,  1.0148779e+00 , -2.8851517e+01],
    # [-1.9166087e-06,   8.1537620e-06 ,  1.0000000e+00]]     #bike
    
    # Hlist1_2 = [[ 1,0,0],[0,1,0],[0,0,1]]     #ubc
    
    Hlist1_2 = [[0.7882767153207999,  0.010905680735846527,  28.170495497465602  ],
    [-0.02537010994777608,  0.9232684706505401,  44.20085016989556 ],
    [-1.1457814415224265E-4,  1.288160474307972E-5,  1.0  ]]     #wall
    
    recall,precision,F = run_gms(img1,img2,Hlist1_2)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F_score:',F)
    