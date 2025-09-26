# this file is to construct a topological network and reconnected network

import math
import cv2
import numpy as np
from scipy.spatial import Delaunay
import judgecross


def getdel(kp1, good):
    # get delaunary
    points = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points[i, 0] = kp1[m.queryIdx].pt[0]
        points[i, 1] = kp1[m.queryIdx].pt[1]
        i = i + 1
    #建立三角剖分网络
    tri = Delaunay(points)
    #print(tri)
    #print(tri.simplices)
    #print(points)
    #返回所有排好序的特征点以及组成的三角形的点的索引
    return [points, tri.simplices]

def getdel2(kp2, good):
    # get delaunary
    points = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points[i, 0] = kp2[m.trainIdx].pt[0]
        points[i, 1] = kp2[m.trainIdx].pt[1]
        i = i + 1
    #建立三角剖分网络
    tri = Delaunay(points)
    #print(tri)
    #print(tri.simplices)
    #print(points)
    #返回所有排好序的特征点以及组成的三角形的点的索引
    return [points, tri.simplices]

#建立拓扑网络并获取所有点的邻居点
def getdel_neighbors1(kp1, good):
    # get delaunary
    points1 = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points1[i, 0] = kp1[m.queryIdx].pt[0]
        points1[i, 1] = kp1[m.queryIdx].pt[1]
        i = i + 1
    #建立三角剖分网络
    tri = Delaunay(points1)
    tri1 = tri.simplices
    #print("tri1:",tri1)
    neighbors1 = []
    for i in range(len(points1)):
        neighbors1.append([])
    for i in range(len(tri1)):
        #获取每个拓扑三角形的点
        t0 = tri1[i][0]
        t1 = tri1[i][1]
        t2 = tri1[i][2]
        if t1 not in neighbors1[t0]:
            neighbors1[t0].append(t1)
        if t2 not in neighbors1[t0]:
            neighbors1[t0].append(t2)
        if t0 not in neighbors1[t1]:
            neighbors1[t1].append(t0)
        if t2 not in neighbors1[t1]:
            neighbors1[t1].append(t2)
        if t0 not in neighbors1[t2]:
            neighbors1[t2].append(t0)
        if t1 not in neighbors1[t2]:
            neighbors1[t2].append(t1)
    return [points1,neighbors1]
def getdel_neighbors2(kp2, good):
    # get delaunary
    points2 = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points2[i, 0] = kp2[m.trainIdx].pt[0]
        points2[i, 1] = kp2[m.trainIdx].pt[1]
        i = i + 1
    #建立三角剖分网络
    tri = Delaunay(points2)
    #print(tri.simplices)
    tri2 = tri.simplices
    neighbors2 = []
    for i in range(len(points2)):
        neighbors2.append([])
    for i in range(len(tri2)):
        t0 = tri2[i][0]
        t1 = tri2[i][1]
        t2 = tri2[i][2]
        if t1 not in neighbors2[t0]:
            neighbors2[t0].append(t1)
        if t2 not in neighbors2[t0]:
            neighbors2[t0].append(t2)
        if t0 not in neighbors2[t1]:
            neighbors2[t1].append(t0)
        if t2 not in neighbors2[t1]:
            neighbors2[t1].append(t2)
        if t0 not in neighbors2[t2]:
            neighbors2[t2].append(t0)
        if t1 not in neighbors2[t2]:
            neighbors2[t2].append(t1)
    return [points2,neighbors2]

#判断一对匹配点的左右拓扑是否相似
def similar_topo(points1,points2,neighbors1,neighbors2):
    #print(len(neighbors1),len(neighbors2))
    good_matches = []

    good =[]
    for i in range(len(neighbors1)):
        real = 0 
        faked = 0
        #判断左边点的邻居点是否匹配右边点的邻居点
        ne1 = neighbors1[i]
        ne2 = neighbors2[i]

        for j in range(len(ne1)):
            if ne1[j] in ne2:
                real = real + 1
            else:
                faked = faked + 2
        #计算得分
        s = real - 1*faked
        if s > 0 :
            good.append([i,s])
            #good_matches.append(i)     
    good.sort(key=lambda x: x[1], reverse=True)
    
    # #消融实验打开
    # for i in range(len(good)):
    #     good_matches.append(good[i][0])
    
    #消融实验关闭    
    for i in range(0,min(len(good),15)):
        #print(good[i])
        good_matches.append(good[i][0])   
    return good_matches



#将keypoints按照matches顺序排列
def changetype(kp2, good):
    points = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points[i, 0] = kp2[m.trainIdx].pt[0]
        points[i, 1] = kp2[m.trainIdx].pt[1]
        i = i + 1
    return points

# draw the topology
def drawdel(points, sim, img1):
    trinum = np.shape(sim)[0]
    h1, w1 = img1.shape[:2]
    view = np.zeros((h1, w1, 3), np.uint8)
    view[:, :, 0] = img1
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    for i in range(0, trinum):
        cv2.line(view, (int(points[sim[i][0]][0]), int(points[sim[i][0]][1])),
                 (int(points[sim[i][1]][0]), int(points[sim[i][1]][1])), (0, 180, 180), 2)
        cv2.line(view, (int(points[sim[i][0]][0]), int(points[sim[i][0]][1])),
                 (int(points[sim[i][2]][0]), int(points[sim[i][2]][1])), (0, 180, 180), 2)
        cv2.line(view, (int(points[sim[i][1]][0]), int(points[sim[i][1]][1])),
                 (int(points[sim[i][2]][0]), int(points[sim[i][2]][1])), (0, 180, 180), 2)
    #cv2.imshow('tri', view)
    view = cv2.resize(view,(int(0.5*view.shape[1]),int(0.5*view.shape[0])))
    return view
    # cv2.waitKey(0)


