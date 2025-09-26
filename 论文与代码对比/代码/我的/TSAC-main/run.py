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
    before,count_before = testmodel.testscoreimage(good, kp1, kp2, H, img1, img2)
    # cv2.imshow("before",before)
    # print("all matches:",count_before)

    if len(good) < 4:
        return None,None,None
    #建立拓扑网络
    points1,neighbors1 = topo.getdel_neighbors1(kp1,good)
    points2,neighbors2 = topo.getdel_neighbors2(kp2,good)

    #计算左右拓扑是否相似
    final = topo.similar_topo(points1,points2,neighbors1,neighbors2)

    final_matches = []
    for f in final:
        final_matches.append(good[f])
        
    out1 = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None ,matchColor = (0,250,0) ,singlePointColor=None, flags=2)
    out1 = cv2.resize(out1,(int(out1.shape[1]*0.5),int(out1.shape[0]*0.5)))
    #cv2.imshow("final",out1)
    print("numbers of final matches: ", len(final_matches))

    
    # #获取按matches排序好的点对
    # newkp1, newkp2 = getmatchingpoints.getnewkp(good, kp1, kp2)
    # # #论文方法过滤匹配点对
    # afterm,newkp1,newkp2 = TSAC.My_ransac(good, newkp1, newkp2, kp1, kp2,final)
    # if len(afterm) ==0:
    #     return None,None,None
    # #通过H矩阵计算正确的匹配点对
    # after,count_after = testmodel.testscoreimage(afterm, kp1, kp2, H,img1,img2)

    # recall = count_after/count_before
    # precision = count_after/len(afterm)
    # F_score = 2*recall*precision/(recall+precision)

    # return recall,precision,F_score
    
    # 消融实验部分
    if len(final_matches) ==0:
         return None,None,None
    after,count_after = testmodel.testscoreimage(final_matches, kp1, kp2, H,img1,img2)
    
    # cv2.imshow("after",after)

    recall = count_after/count_before
    precision = count_after/len(final)
    F_score = 2*recall*precision/(recall+precision)
    
    # cv2.waitKey(0)
    return recall,precision,F_score



if __name__ == '__main__':
    img1= "E:\paper_testcode\match\dataset\\test\\v_yard\\1.ppm"    
    img2= "E:\paper_testcode\match\dataset\\test\\v_yard\\3.ppm"
    Hlist1_3 = [[0.81916, -0.19602, -221.9],[-0.18509, 0.99057, -55.575],[-0.00033371, -0.00011673, 0.99377]] 
    recall,precision,F_score=run(img1,img2,Hlist1_3)
    print(recall,precision,F_score)
    