import cv2
import math
import numpy as np
import getmatchpoints
import testmodel

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

# generate a median KNN graph for the given point set
def generate_median_KNN_graph(K, distance_matrix, match_number):
    # initialize the adjacency matrix Ap or Ap'
    adj_matrix = np.zeros((match_number, match_number))
    # calculate the median distance \eta
    dis_median = np.median(distance_matrix)
    for i in range(0, match_number):
        # search the K nearest neighbors of the i-th node vi
        K_idx, K_dis, sorted_idx = find_KNN(K, ori_dis_mat, i)
        for j, dis in zip(K_idx, K_dis):
            # if j in the K nearest neighbors list and the distance between vi and vj is less than \eta
            if dis <= dis_median:
                # connect vi and vj with non-direction edge
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    # return the adjacency matrix Ap or Ap'
    return adj_matrix

# find the outlier
def find_outlier(ori_adj_matrix, dst_adj_matrix):
    # calculate the residual adjacency matrix R
    residual_adj_mat = abs(ori_adj_matrix - dst_adj_matrix)
    # find the outlier column that yields the maximal number of different edges in both graphs
    outlier_j = np.argmax(np.sum(residual_adj_mat, axis=1))
    # return the residual adjacency matrix R and the index of outlier column
    return residual_adj_mat, outlier_j

# remove the outlier from the distance matrix and point set
def remove_outlier(distance_matrix, cord_list, outlier_j):
    update_dis_matrix = np.delete(distance_matrix, outlier_j, axis=0)
    update_dis_matrix = np.delete(update_dis_matrix, outlier_j, axis=1)
    update_cord_list = np.delete(cord_list, outlier_j, axis=0)
    return update_dis_matrix, update_cord_list

# ori_img =cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\v_yuri\\1.ppm', 0)
# dst_img =cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\v_yuri\\3.ppm', 0)

ori_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\i_yellowtent\\1.ppm', 0)
dst_img = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\i_yellowtent\\3.ppm', 0)

result = getmatchpoints.siftandgetlocation(ori_img, dst_img,1)

bf_matches = result[0]
ori_kp = result[1]
dst_kp = result[2]

Hlist1_3 = [[1,0, 0],[0, 1, 0],[0, 0, 1]] 
# Hlist1_3 = [[0.80989,0.15593, -274.24],[-0.19006, 1.3082, -274.15],[-0.00047728, 0.0001622, 1]]                                         #v_yuri
# Hlist1_3 = [[0.81916, -0.19602, -221.9],[-0.18509, 0.99057, -55.575],[-0.00033371, -0.00011673, 0.99377]]                                #v_yard
H = np.array(Hlist1_3)

print('the correct matches before removal:')
before,count_before = testmodel.testscoreimage(bf_matches, ori_kp, dst_kp, H, ori_img, dst_img)
outimg=cv2.drawMatches(ori_img,ori_kp,dst_img,dst_kp,bf_matches,None,flags=2)
cv2.imshow("before",before)

ori_pt_list = []  # create a list of original matching points
dst_pt_list = []  # create a list of destination matching points
filter_matches = [] # create a list of matches
for m in bf_matches:
    ori_pt_x = ori_kp[m.queryIdx].pt[0]  # get abscissas of original matching points
    ori_pt_y = ori_kp[m.queryIdx].pt[1]  # get ordinates of original matching points

    dst_pt_x = dst_kp[m.trainIdx].pt[0]  # get abscissas of destination matching points
    dst_pt_y = dst_kp[m.trainIdx].pt[1]  # get ordinates of destination matching points

    # filter out duplicate matching points
    if ([ori_pt_x, ori_pt_y] not in ori_pt_list) and ([dst_pt_x, dst_pt_y] not in dst_pt_list):
        ori_pt_list.append([ori_pt_x, ori_pt_y])
        dst_pt_list.append([dst_pt_x, dst_pt_y])
        filter_matches.append(m)

# extract the coordinates of original match points
# ori_cord_list = np.float32([ori_kp[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
ori_cord_list = np.float32(ori_pt_list).reshape(-1, 1, 2)
# extract the coordinates of destination match points
# dst_cord_list = np.float32([dst_kp[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
dst_cord_list = np.float32(dst_pt_list).reshape(-1, 1, 2)


# calculate the distance matrix of original match points
ori_dis_mat = distance_cal(ori_cord_list)
# calculate the distance matrix of destination match points
dst_dis_mat = distance_cal(dst_cord_list)

match_number = len(ori_cord_list)  # the number of initial match points, N
while True:
    K = 5  # the number of nearest neighbors
    ori_adj_matrix = generate_median_KNN_graph(K, ori_dis_mat, match_number)  # the adjacency matrix of ori_graph, Ap
    dst_adj_matrix = generate_median_KNN_graph(K, dst_dis_mat, match_number)  # the adjacency matrix of dst_graph, Ap'
    # the residual adjacency matrix and the outlier index, R and j^out
    residual_adj_mat, outlier_j = find_outlier(ori_adj_matrix, dst_adj_matrix)
    print(np.sum(residual_adj_mat))
    # if the residual adjacency matrix is a zero matrix
    if np.all(residual_adj_mat == 0):
        # stop iteration
        break
    # update the ori_dis_mat and ori_cord_list
    ori_dis_mat, ori_cord_list = remove_outlier(ori_dis_mat, ori_cord_list, outlier_j)
    # update the dst_dis_mat and dst_cord_list
    dst_dis_mat, dst_cord_list = remove_outlier(dst_dis_mat, dst_cord_list, outlier_j)
    match_number -= 1  # the number of match points is decreased by one
    filter_matches.pop(outlier_j) # remove the outlier index from the index list

# get the final matches
final_match = filter_matches
print(len(final_match)) # print
final_result = cv2.drawMatches(ori_img, ori_kp, dst_img, dst_kp, final_match, None,matchColor = (0,240,0) ,singlePointColor=None, flags=2)
cv2.imshow('final_result_GTM', final_result)

#通过H矩阵计算正确的匹配点对
summ = testmodel.testscore(final_match, ori_kp, dst_kp, H)
print('the number of correct matches after removal:', summ)
after,count_after = testmodel.testscoreimage(final_match, ori_kp, dst_kp, H,ori_img,dst_img)
cv2.imshow('after', after)

# recall = count_after/count_before
# precision = count_after/len(final_match)
# F_score = 2*recall*precision/(recall+precision)

# print("召回率：",recall)
# print("准确率：",precision)
# print("F-score:",F_score)

cv2.waitKey()






