import numpy as np

#表示两个向量的叉积，可以用于判断这三个点的相对位置关系。如果返回值大于 0，则说明这三个点按逆时针方向排列；如果小于 0，则说明按顺时针方向排列；如果等于 0，则说明这三个点共线。
def cross(p1,p2,p3): 
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1

#判断两条线段是否相交,1表示相交
def judge(line1a, line1b, line2c, line2d):
    if max(line1a[0], line1b[0]) <= min(line2c[0], line2d[0]):
        return 0
    elif max(line2c[0], line2d[0]) <= min(line1a[0], line1b[0]):
        return 0
    elif max(line1a[1], line1b[1]) <= min(line2c[1], line2d[1]):
        return 0
    elif max(line2c[1], line2d[1]) <= min(line1a[1], line1b[1]):
        return 0
    else:
        
        result1 = cross(line1a,line1b,line2c)*cross(line1a,line1b,line2d)
        result2 = cross(line2c,line2d,line1a)*cross(line2c,line2d,line1b)
        if result2 < 0 and result1 < 0:
            return 1
        else:
            return 0
