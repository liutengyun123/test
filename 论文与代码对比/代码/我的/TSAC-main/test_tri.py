import numpy as np  
import matplotlib.pyplot as plt  
from scipy.spatial import Delaunay  


def draw(kps,good):
    # 输入二维点  
    points = np.zeros((len(good), 2))
    i=0
    for m in good:
        points[i, 0] = kps[m.queryIdx].pt[0]
        points[i, 1] = kps[m.queryIdx].pt[1]
        i = i + 1 
    
    # 构建三角剖分  
    tri = Delaunay(points)  
    
    # 绘制三角剖分图  
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)  
    plt.plot(points[:, 0], points[:, 1], 'o')  
    plt.title('Delaunay Triangulation')  
    plt.show()

def draw2(kps,good):
    # 输入二维点  
    points = np.zeros((len(good), 2))
    i=0
    for m in good:
        points[i, 0] = kps[m.trainIdx].pt[0]
        points[i, 1] = kps[m.trainIdx].pt[1]
        i = i + 1 
    
    # 构建三角剖分  
    tri = Delaunay(points)  
    
    # 绘制三角剖分图  
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)  
    plt.plot(points[:, 0], points[:, 1], 'o')  
    plt.title('Delaunay Triangulation')  
    plt.show()