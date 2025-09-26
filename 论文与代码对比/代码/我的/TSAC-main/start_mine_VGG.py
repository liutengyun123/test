import cv2
import numpy as np
import getmatchingpoints
import testmodel
import topo
import TSAC
import test_tri

"""
实际上该论文的主要思想是：
1. 先用构建好的拓扑三角结构计算出来的相交概率计算最佳的before_H，利用before_H过滤匹配点对
2. 再使用RANSAC再次过滤匹配点对
"""
#"""VGG数据集测试"""

# img1 = cv2.imread('E:\paper_testcode\match\\VGG\\bark\\img1.ppm', 0)
# img2 = cv2.imread('E:\paper_testcode\match\\VGG\\bark\\img2.ppm', 0)

img1 = cv2.imread('E:\paper_testcode\match\\VGG\\bikes\\img1.ppm', 0)
img2 = cv2.imread('E:\paper_testcode\match\\VGG\\bikes\\img2.ppm', 0)

# img1 = cv2.imread('E:\paper_testcode\match\\VGG\\ubc\\img1.ppm', 0)
# img2 = cv2.imread('E:\paper_testcode\match\\VGG\\ubc\\img2.ppm', 0)

# img1 = cv2.imread('E:\paper_testcode\match\\VGG\\wall\\img1.ppm', 0)
# img2 = cv2.imread('E:\paper_testcode\match\\VGG\\wall\\img2.ppm', 0)

# Hlist1_2 = [[0.7022029025774007,  0.4313737491020563,  -127.94661199701689],
#             [-0.42757325092889575,  0.6997834349758094,  201.26193857481698],
#             [4.083733373964227E-6,  1.5076445750988132E-5,  1.0]]     #bark
    
Hlist1_2 = [[   1.0107879e+00,   8.2814684e-03,   1.8576800e+01],
  [-4.9128885e-03 ,  1.0148779e+00 , -2.8851517e+01],
  [-1.9166087e-06,   8.1537620e-06 ,  1.0000000e+00]]     #bike

# Hlist1_2 = [[ 1,0,0],[0,1,0],[0,0,1]]     #ubc

# Hlist1_2 = [[0.7882767153207999,  0.010905680735846527,  28.170495497465602  ],
#   [-0.02537010994777608,  0.9232684706505401,  44.20085016989556 ],
#   [-1.1457814415224265E-4,  1.288160474307972E-5,  1.0  ]]     #wall

    
H = np.array(Hlist1_2)

#获取good_matches,keypoints1,keypoints2
result = getmatchingpoints.siftandgetlocation(img1, img2, 1)

good = result[0]
kp1 = result[1]
kp2 = result[2]
print('the correct matches before removal:')
before,count_before = testmodel.testscoreimage(good, kp1, kp2, H, img1, img2)
cv2.imshow("before",before)
out1 = cv2.drawMatches(img1, kp1, img2, kp2, good, None ,matchColor = (0,250,0) ,singlePointColor=None, flags=2)
out1 = cv2.resize(out1,(int(out1.shape[1]),int(out1.shape[0])))


#建立拓扑网络
points1,neighbors1 = topo.getdel_neighbors1(kp1,good)
points2,neighbors2 = topo.getdel_neighbors2(kp2,good)

#显示拓扑图
cv2.imshow("before_TASC", out1)
#test_tri.draw(kp1,good)
#test_tri.draw2(kp2,good)

#计算左右拓扑是否相似
final = topo.similar_topo(points1,points2,neighbors1,neighbors2)

final_matches = []
for f in final:
    final_matches.append(good[f])
print("mine:",len(final_matches))

# *************************************************非消融实验部分
#获取按matches排序好的点对
newkp1, newkp2 = getmatchingpoints.getnewkp(good, kp1, kp2)
# #论文方法过滤匹配点对
afterm,newkp1,newkp2 = TSAC.My_ransac(good, newkp1, newkp2, kp1, kp2,final)
print('the number of matches after removal:', len(afterm))

#getmatchingpoints.showtheimage(afterm,newkp1,newkp2,img1,img2)
out = cv2.drawMatches(img1, kp1, img2, kp2, afterm, None ,matchColor = (0,240,0) ,singlePointColor=None, flags=2)
out = cv2.resize(out,(int(0.5*out.shape[1]),int(0.5*out.shape[0])))
cv2.imshow('my_TASC', out)

#通过H矩阵计算正确的匹配点对
summ = testmodel.testscore(afterm, kp1, kp2, H)
print('the number of correct matches after removal:', summ)
after,count_after = testmodel.testscoreimage(afterm, kp1, kp2, H,img1,img2)
cv2.imshow('after', after)

recall = count_after/count_before
precision = count_after/len(afterm)
F_score = 2*recall*precision/(recall+precision)

# 消融实验部分
# after,count_after = testmodel.testscoreimage(final_matches, kp1, kp2, H,img1,img2)

# cv2.imshow("after",after)

# recall = count_after/count_before
# precision = count_after/len(final)
# F_score = 2*recall*precision/(recall+precision)

print("召回率：",recall)
print("准确率：",precision)
print("F-score:",F_score)


cv2.waitKey(0)