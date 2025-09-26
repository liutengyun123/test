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


ori_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\v_yuri\\1.ppm', 0)
dst_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\v_yuri\\3.ppm', 0)

# ori_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\i_yellowtent\\1.ppm', 0)
# dst_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\i_yellowtent\\3.ppm', 0)

result = siftandgetlocation(ori_img, dst_img, 1)
matches = result[0]
ori_kp = result[1]
dst_kp = result[2]

#Hlist1_3 = [[1,0, 0],[0, 1, 0],[0, 0, 1]]                                                                                      #i_yellowtent
#Hlist1_3 = [[0.81916, -0.19602, -221.9],[-0.18509, 0.99057, -55.575],[-0.00033371, -0.00011673, 0.99377]]
Hlist1_3 = [[0.80989,0.15593, -274.24],[-0.19006, 1.3082, -274.15],[-0.00047728, 0.0001622, 1]]                              #v_yuri 1-4
Hlist1_4 = [[2.4144,-0.0022023,-199.3],[0.52146,2.0547,-569.49],[0.0010423,8.4489e-05,1.0043]]                                #v_london   1_4
H = np.array(Hlist1_3)


print('the correct matches before removal:')
before,count_before = testmodel.testscoreimage(matches, ori_kp, dst_kp, H, ori_img, dst_img)
# 显示结果
cv2.imshow('before', before)

# 计算匹配点的数量
num_matches = len(matches)
print("特征匹配点数量：", num_matches)


#使用gms剔除误匹配
matches_gms = cv2.xfeatures2d.matchGMS(ori_img.shape[:2], ori_img.shape[:2], ori_kp, dst_kp, matches, withScale=False, withRotation=False,thresholdFactor=6)
out_gms = cv2.drawMatches(ori_img,ori_kp,dst_img,dst_kp,matches_gms,None,matchColor = (0,240,0) ,singlePointColor=None, flags=2)
cv2.imshow('gms', out_gms)

final_matches = []
for m in matches_gms:
    final_matches.append(m)

print(len(final_matches),len(matches_gms))
#通过H矩阵计算正确的匹配点对
summ = testmodel.testscore(final_matches, ori_kp, dst_kp, H)
print('the number of correct matches after removal:', summ)
after,count_after = testmodel.testscoreimage(final_matches, ori_kp, dst_kp, H,ori_img,dst_img)
cv2.imshow('after', after)

recall = count_after/count_before
precision = count_after/len(final_matches)
F_score = 2*recall*precision/(recall+precision)

print("召回率：",recall)
print("准确率：",precision)
print("F-score:",F_score)

cv2.waitKey()