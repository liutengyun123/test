import cv2
import math
import numpy as np
import getmatchpoints
from enum import Enum
import testmodel


class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5
 
 
def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
 
    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))
 
    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
 
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)
 
    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)
 
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
 
            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient
 
            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


# calculate the distance between any two nodes in the key point set
def distance_cal(cord_list):
    num = len(cord_list)  # the number of node
    distance_matrix = np.zeros([num, num])  # the distance matrix
    for i, cord_i in enumerate(cord_list):
        for j, cord_j in enumerate(cord_list):
            # calculate the Euclidean distance
            distance = math.sqrt((cord_i[0][0] - cord_j[0][0]) ** 2 + (cord_i[0][1] - cord_j[0][1]) ** 2)
            if i != j:
                # avoid zero distance except for diagonal elements
                distance_matrix[i][j] = distance + 0.0000001
        # print(distance_matrix[i])
    return distance_matrix

# search K nearest neighbors of the i-th node
def find_KNN(K, distance_matrix, index):
    distance_vector = np.array(distance_matrix[index])  # extract the distance vector of the i-th node
    # get an enumeration index vector ranging from 0 to the number of nodes
    index_vector = np.arange(0, len(distance_vector))
    sorted_dis_vec = np.lexsort((index_vector.T, distance_vector.T))  # sort by distance in ascending order
    # print(sorted_dis_vec)
    # get the indexes of K nearest neighbors of the i-th node and the first one is itself
    K_neighbor_index = sorted_dis_vec[:K+1]
    # get the distances of K nearest neighbors of the i-th node and the first one is 0
    K_neighbor_distance = distance_vector[K_neighbor_index]
    return K_neighbor_index, K_neighbor_distance, sorted_dis_vec

# calculate the ranking shift of the i-th node
def cal_ranking_shift(K_idx_list, sorted_idx_list):
    phi_idx = []  # a binary sequence phi_vi of the i-th node vi
    # Iterate over the K nearest neighbors of the i-th node
    for k, n in enumerate(K_idx_list):
        # n is the k-th nearest neighbor vj of the node vi
        # r is the ranking of n's corresponding node v'j centered on v'i
        r = list(sorted_idx_list).index(n)
        phi = r > k  # if i > k, phi = 1 else phi = 0
        phi_idx.append(phi)
    return phi_idx

# calculate the node affinity score s(vi, v'i)
def cal_node_affinity_score(ori_phi_idx, dst_phi_idx, K):
    # convert the binary sequences phi_vi and phi_v'i to numpy arrays
    ori_phi_idx = np.array(ori_phi_idx)
    dst_phi_idx = np.array(dst_phi_idx)
    # s(vi, v'i) = 1 - 1/2K * |phi_vi + phi_v'i|
    node_s_idx = 1 - np.sum(abs(np.add(ori_phi_idx, dst_phi_idx))) / (2 * K)
    return node_s_idx

# calculate the edge affinity score s(e_iij, e'_iij)
def cal_edge_affinity_score(idx, x_idx, ori_K_idx, dst_K_idx, ori_dis_mat, dst_dis_mat, K):
    edge_s_idx_list = []
    # Iterate over the K nearest neighbors of the i-th node expect for itself
    for j in range(len(ori_K_idx[1:])):
        # if x_idx[n] == 1:
        # i_j is the j-th neighbor of node vi
        i_j = ori_K_idx[j+1]
        # get the edge distance d(e_iij) between node vi and its j-th neighbor node vij
        ori_edge_dis = ori_dis_mat[idx][i_j]
        # i_j_ is the j-th neighbor of corresponding node v'i
        i_j_ = dst_K_idx[j+1]
        # get the edge distance d(e'_iij) between corresponding node v'i and its j-th neighbor node v'ij
        dst_edge_dis = dst_dis_mat[idx][i_j_]
        # print(idx, i_j, ori_edge_dis, i_j_,des_edge_dis)
        # s(e_iij, e'_iij) = 1/k * exp(-|d(e_iij) - d(e'_iij)| / max(d(e_iij), d(e'_iij)))
        edge_s_idx = np.exp(- abs(ori_edge_dis - dst_edge_dis) / max(ori_edge_dis, dst_edge_dis)) / K
        # print(edge_s_idx)
        edge_s_idx_list.append(edge_s_idx)

    return edge_s_idx_list


def run_lgsc(img1,img2,H):  
    ori_img =cv2.imread(img1, 0)
    dst_img =cv2.imread(img2, 0)

    result = getmatchpoints.siftandgetlocation(ori_img, dst_img,1)

    bf_matches = result[0]
    ori_kp = result[1]
    dst_kp = result[2]
    
    if len(bf_matches) < 4:
        return None,None,None
    
    H = np.array(H)

    before,count_before = testmodel.testscoreimage(bf_matches, ori_kp, dst_kp, H, ori_img, dst_img)

    # extract the coordinates of original match points
    # 将匹配点对转化为float类型的numpy数组
    ori_cord_list = np.float32([ori_kp[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
    # extract the coordinates of destination match points
    dst_cord_list = np.float32([dst_kp[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)

    # calculate the distance matrix of original match points
    # 计算匹配点对的距离列表
    ori_dis_mat = distance_cal(ori_cord_list)
    # calculate the distance matrix of destination match points
    dst_dis_mat = distance_cal(dst_cord_list)

    count =0
    iter_num = 2  # the iterate number
    for iter in range(iter_num):
        ini_match_number = len(ori_cord_list)  # the number of initial match points
        #print(ini_match_number)
        K_list = [7, 5]  # the set of K
        lambda_ = 0.7  # the parameter lambda
        flitered_match = []  # the list of filtered match points
        # iterate over the initial match points
        for idx in range(0, ini_match_number):
            s_idx = 0  # the match score of the i-th node
            # iterate over the set of K
            for K in K_list:
                # search the K nearest neighbors of the i-th node vi
                ori_K_idx, ori_K_dis, sorted_ori_idx = find_KNN(K, ori_dis_mat, idx)
                # search the K nearest neighbors of the corresponding node v'i
                dst_K_idx, dst_K_dis, sorted_dst_idx = find_KNN(K, dst_dis_mat, idx)
                # print(ori_K_idx, des_K_idx)
                # construct the local graph
                # if the neighbor of node vi is also in the corresponding node v'i neighborhoods, x_i = 1, else x_i = 0
                x_idx = [1 if i in dst_K_idx else 0 for i, j in zip(ori_K_idx, dst_K_idx)]
                # calculate the ranking shift of the i-th node vi
                ori_phi_idx = cal_ranking_shift(ori_K_idx, sorted_dst_idx)
                # calculate the ranking shift of the corresponding node v'i
                dst_phi_idx = cal_ranking_shift(dst_K_idx, sorted_ori_idx)
                # calculate the node affinity score of nodes vi and v'i
                node_s_idx = cal_node_affinity_score(ori_phi_idx, dst_phi_idx, K)
                # calculate the edge affinity score of nodes vi and v'i
                edge_s_idx_list = cal_edge_affinity_score(idx, x_idx, ori_K_idx, dst_K_idx, ori_dis_mat, dst_dis_mat, K)
                edge_s_idx_list.insert(0, node_s_idx)
                # construct the local affinity vector w_i
                w_idx = edge_s_idx_list
                # calculate the match score s_i = w_i * x_i
                s_idx_K = np.matmul(np.array(w_idx), np.array(x_idx).T)
                # accumulate the match score
                s_idx += s_idx_K
            # calculate the mean match score among different K
            s_idx = s_idx / len(K_list)
            # if the match score is greater than the preset threshold lambda
            if s_idx > lambda_:
                # the match node vi will be remained
                flitered_match.append(idx)

        # update the original match points
        ori_cord_list = ori_cord_list[flitered_match]
        # update the destination match points
        dst_cord_list = dst_cord_list[flitered_match]

    final_match = [bf_matches[i] for i in flitered_match]

    #通过H矩阵计算正确的匹配点对
    after,count_after = testmodel.testscoreimage(final_match, ori_kp, dst_kp, H,ori_img,dst_img)

    if count_before ==0 or len(final_match) == 0:
        return None,None,None
    recall = count_after/count_before
    precision = count_after/len(final_match)
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
    
    recall,precision,F = run_lgsc(img1,img2,Hlist1_2)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F_score:',F)
    

