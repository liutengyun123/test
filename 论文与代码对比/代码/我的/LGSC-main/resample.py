# this file contain the sampling
import cv2
import numpy as np
import math
import random

import testmodel


def RANSACUpdateNumIters(p, ep,model_points, max_iters):
    if model_points <= 0:
        print("the number of model points should be positive")
    p = max(p, 0.)
    p = min(p, 1.)
    ep = max(ep, 0.)
    ep = min(ep, 1.)

    num = max(1.0 - p, 0.0001)
    denom = 1. - pow(1. - ep, model_points)
    if denom < 0.0001:
        return 0

    num = math.log(num)
    denom = math.log(denom)
    if denom >= 0 or -num >= max_iters * (-denom):
        return max_iters
    else:
        return round(num/denom)

def findqueryidx(listindex,M):
    N=[]
    for m in M:
        for index in listindex:
            if index[1]==m:
                N.append(index[0])
    if len(N)==4:
        return N
    else:
        print('find wrong in index')
        return M

def My_ransac(good, kp1, kp2,oldkp1,oldkp2,lst):
    """
    先用构建好的拓扑三角结构计算出来的相交概率计算最佳的before_H矩阵，
    再根据before_H矩阵过滤误匹配得到过滤匹配点对
    最后在过滤匹配点对中计算H矩阵，再次利用H矩阵过滤匹配点对
    """
    time = 0
    maxcount = 0
    rightH = []

    indexlist=[]
    '''for m in good:
        indexlist.append([m.queryIdx,m.trainIdx])'''
    maxiter = 2000 # in OpenCV, the default value of maxiter is 1000
    while True:
        time=time+1
        #获得四个采样点对
        if len(lst)<4:
            return {},{},{}
        M = random.sample(lst,4)
        # N = findqueryidx(indexlist, M)
        loc1 = np.array([[[kp1[M[0]].pt[0], kp1[M[0]].pt[1]]],
                         [[kp1[M[1]].pt[0], kp1[M[1]].pt[1]]],
                         [[kp1[M[2]].pt[0], kp1[M[2]].pt[1]]],
                         [[kp1[M[3]].pt[0], kp1[M[3]].pt[1]]]])
        loc2 = np.array([[[kp2[M[0]].pt[0], kp2[M[0]].pt[1]]],
                         [[kp2[M[1]].pt[0], kp2[M[1]].pt[1]]],
                         [[kp2[M[2]].pt[0], kp2[M[2]].pt[1]]],
                         [[kp2[M[3]].pt[0], kp2[M[3]].pt[1]]]])

        #四点法计算H矩阵
        H,_ = cv2.findHomography(loc1, loc2)
        if H is None:
            continue
        count = 0
        for m in good:
            loc1 = [oldkp1[m.queryIdx].pt[0], oldkp1[m.queryIdx].pt[1]]
            loc2 = [oldkp2[m.trainIdx].pt[0], oldkp2[m.trainIdx].pt[1]]
            result = testmodel.basetest(loc1, loc2, H)
            count = count+result
        if count > maxcount:
            maxcount = count
            # print('符合条件点数更新：',maxcount)
            rightH = H
        # ep = 1 - count / len(good)
        #maxiter1 = RANSACUpdateNumIters(0.999, ep, 1, maxiter)
        #print("maxiter1:",maxiter1)
        if time >= maxiter:
            # print('100次内最佳结果：',rightH)
            break
    afterm = []
    for m in good:
        loc1 = [oldkp1[m.queryIdx].pt[0], oldkp1[m.queryIdx].pt[1]]
        loc2 = [oldkp2[m.trainIdx].pt[0], oldkp2[m.trainIdx].pt[1]]
        result = testmodel.basetest(loc1, loc2, rightH)
        if result == 1:
            afterm.append(m)
    
    #直接用RANSAC计算H矩阵
    src_pts = np.float32([oldkp1[m.queryIdx].pt for m in afterm]).reshape(-1, 1, 2)
    dst_pts = np.float32([oldkp2[m.trainIdx].pt for m in afterm]).reshape(-1, 1, 2)
    Hnew, _ = cv2.findHomography(src_pts, dst_pts, 0, ransacReprojThreshold=3)
    newafterm = []
    newkp1 = []
    newkp2 = []
    for m in good:
        loc1 = [oldkp1[m.queryIdx].pt[0], oldkp1[m.queryIdx].pt[1]]
        loc2 = [oldkp2[m.trainIdx].pt[0], oldkp2[m.trainIdx].pt[1]]
        result = testmodel.basetest(loc1, loc2, Hnew)
        if result == 1:
            newafterm.append(m)
            newkp1.append(oldkp1[m.queryIdx])
            newkp2.append(oldkp2[m.trainIdx])
    # print(type(newafterm),type(newkp1),type(newkp2))
    return newafterm, newkp1, newkp2

