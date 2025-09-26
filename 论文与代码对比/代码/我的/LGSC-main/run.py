import cv2
import numpy as np
import getmatchpoints
import testmodel


def run(image1,image2,H):
    ori_img = cv2.imread(image1, 0)
    dst_img = cv2.imread(image2, 0)

    H = np.array(H)
 
    #获取good_matches,keypoints1,keypoints2
    result = getmatchpoints.siftandgetlocation(ori_img, dst_img, 1)

    matches = result[0]
    ori_kp = result[1]
    dst_kp = result[2]

    if len(matches) < 4:
        return None,None,None

    before,count_before = testmodel.testscoreimage(matches, ori_kp, dst_kp, H, ori_img, dst_img)
    
    cv2.imshow("before",before)

    # 计算匹配点的数量
    num_matches = len(matches)

    # 使用RANSAC算法进行误匹配的剔除
    src_pts = np.float32([ori_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC,5)
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_PROSAC,5)
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)

    # 剔除误匹配的特征点
    good_matches = [matches[i] for i, m in enumerate(mask) if m]

    #通过H矩阵计算正确的匹配点对
    summ = testmodel.testscore(good_matches, ori_kp, dst_kp, H)
    after,count_after = testmodel.testscoreimage(good_matches, ori_kp, dst_kp, H,ori_img,dst_img)

    if count_before ==0 or len(good_matches) == 0:
        return None,None,None
    recall = count_after/count_before
    precision = count_after/len(good_matches)
    if(recall + precision == 0):
        return None,None,None
    F_score = 2*recall*precision/(recall+precision)
    
    cv2.imshow('after', after)
    cv2.waitKey()

    return recall,precision,F_score

if __name__ == '__main__':
    img1 = 'E:\paper_testcode\match\\VGG\\bikes\\img1.ppm'
    img2 = 'E:\paper_testcode\match\\VGG\\bikes\\img2.ppm'
    
    # img1 = 'E:\paper_testcode\match\\VGG\\ubc\\img1.ppm'
    # img2 = 'E:\paper_testcode\match\\VGG\\ubc\\img2.ppm'
    
    # img1 = 'E:\paper_testcode\match\\VGG\\wall\\img1.ppm'
    # img2 = 'E:\paper_testcode\match\\VGG\\wall\\img2.ppm'
     
    
    Hlist1_2 = [[   1.0107879e+00,   8.2814684e-03,   1.8576800e+01],
    [-4.9128885e-03 ,  1.0148779e+00 , -2.8851517e+01],
    [-1.9166087e-06,   8.1537620e-06 ,  1.0000000e+00]]     #bike
    
    # Hlist1_2 = [[ 1,0,0],[0,1,0],[0,0,1]]     #ubc
    
    # Hlist1_2 = [[0.7882767153207999,  0.010905680735846527,  28.170495497465602  ],
    # [-0.02537010994777608,  0.9232684706505401,  44.20085016989556 ],
    # [-1.1457814415224265E-4,  1.288160474307972E-5,  1.0  ]]     #wall
    
    recall,precision,F = run(img1,img2,Hlist1_2)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F_score:',F)
    