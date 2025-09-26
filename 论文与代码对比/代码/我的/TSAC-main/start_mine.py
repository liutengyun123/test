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
#"""H_patches数据集测试"""


#img1 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_yard\\1.ppm', 0)
#img2 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_yard\\3.ppm', 0)

# img1 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_strand\\1.ppm', 0)
# img2 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_strand\\3.ppm', 0)

# img1 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_yuri\\1.ppm', 0)
# img2 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_yuri\\3.ppm', 0)

img1 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_vitro\\1.ppm', 0)
img2 = cv2.imread('E:\paper_testcode\match\\v_weapons\\v_vitro\\3.ppm', 0)

# Hlist1_3 = [[0.81916, -0.19602, -221.9],[-0.18509, 0.99057, -55.575],[-0.00033371, -0.00011673, 0.99377]]  #yard
# Hlist1_3 = [[0.73064, -0.15447, 105.22],[-0.058424, 0.44937, 109.29],[-2.5324e-05, -0.00034262, 0.999]]     #strand
# Hlist1_3 = [[0.80989, 0.15593, -274.24],[-0.19006, 1.3082, -274.15],[-0.00047728, 0.0001622, 1]]     #strand
Hlist1_3 = [[1.3849, -0.016724, 142.49],[0.39374, 1.2561, -188.47],[0.00065186, 1.5824e-07, 1]]     #vitro

H = np.array(Hlist1_3)

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


# 消融实验部分
# after,count_after = testmodel.testscoreimage(final_matches, kp1, kp2, H,img1,img2)

# cv2.imshow("after",after)
#************************************************************


recall = count_after/count_before
precision = count_after/len(afterm)
F_score = 2*recall*precision/(recall+precision)

print("召回率：",recall)
print("准确率：",precision)
print("F-score:",F_score)



cv2.waitKey(0)