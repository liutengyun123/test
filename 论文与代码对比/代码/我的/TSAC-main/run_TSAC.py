import cv2
import numpy as np
import getmatchingpoints
import testmodel
import topo
import TSAC
import test_tri

def run(image1,image2,H):
    img1 = cv2.imread(image1, 0)
    img2 = cv2.imread(image2, 0)

    H = np.array(H)

   #获取good_matches,keypoints1,keypoints2
    result = getmatchingpoints.siftandgetlocation(img1, img2, 1)

    good = result[0]
    kp1 = result[1]
    kp2 = result[2]
    # print('the correct matches before removal:')
    before,count_before = testmodel.testscoreimage(good, kp1, kp2, H, img1, img2)
    # cv2.imshow("before",before)
    out1 = cv2.drawMatches(img1, kp1, img2, kp2, good, None ,matchColor = (0,250,0) ,singlePointColor=None, flags=2)
    out1 = cv2.resize(out1,(int(0.5*out1.shape[1]),int(0.5*out1.shape[0])))

    # cv2.imshow("before_TASC", out1)

    #建立拓扑网络
    result1 = topo.getdel(kp1, good)
    points1 = result1[0]
    points2 = topo.changetype(kp2, good)
    sim = result1[1]
    view1 = topo.drawdel(points1, sim, img1)
    view2 = topo.drawdel(points2, sim, img2)

    # cv2.imshow("view1",view1)
    #计算三角拓扑网络内每个点的转换概率
    p = topo.gettimes2(sim, points2)
    #获取按matches排序好的点对
    newkp1, newkp2 = getmatchingpoints.getnewkp(good, kp1, kp2)

    #论文方法过滤匹配点对
    afterm,newkp1,newkp2 = TSAC.ransacp(good, newkp1, newkp2, p, kp1, kp2)
    # print('the number of matches after removal:', len(afterm))
    # print(len(newkp1), len(newkp2),len(afterm))
    #getmatchingpoints.showtheimage(afterm,newkp1,newkp2,img1,img2)
    if afterm==0:
        return 0,0,0
    out = cv2.drawMatches(img1, kp1, img2, kp2, afterm, None ,matchColor = (0,240,0) ,singlePointColor=None, flags=2)
    out = cv2.resize(out,(int(0.5*out.shape[1]),int(0.5*out.shape[0])))
    # cv2.imshow('after_TASC', out)

    # #绘制过滤完的匹配点对

    #通过H矩阵计算正确的匹配点对
    summ = testmodel.testscore(afterm, kp1, kp2, H,)
    # print('the number of correct matches after removal:', summ)
    after,count_after = testmodel.testscoreimage(afterm, kp1, kp2, H,img1,img2)
    # cv2.imshow('after', after)

    recall = count_after/count_before
    precision = count_after/len(afterm)
    F_score = 2*recall*precision/(recall+precision)
    
    # cv2.waitKey(0)
    return recall,precision,F_score



if __name__ == '__main__':
    img1= "E:\paper_testcode\match\dataset\\test\\i_crownnight\\1.ppm"    
    img2= "E:\paper_testcode\match\dataset\\test\\i_crownnight\\3.ppm"
    Hlist1_3 = [[0.81916, -0.19602, -221.9],[-0.18509, 0.99057, -55.575],[-0.00033371, -0.00011673, 0.99377]] 
    # Hlist1_3 =[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    recall,precision,F_score=run(img1,img2,Hlist1_3)
    print(recall,precision,F_score)
    