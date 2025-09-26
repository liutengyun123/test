# this file is to sample the matching points
# according to mismatching probability

import math
import random

import numpy as np

# getp is to get the probability being selected
#根据概率进行归一化得分
def getp(prob):
    thesize = np.shape(prob)[0]
    score = np.zeros(thesize)
    for i in range(0, thesize):
        score[i] = math.pow(1-prob[i],1)
    summ = np.sum(score)
    for i in range(0,thesize):
        score[i]=score[i]/summ
    return score

# sample 4 matching points
#选择四个匹配点
def samplep(score):
    thesize = np.shape(score)[0]
    if thesize<4:
        return [-1,-1,-1,-1]
    fanwei =np.zeros(thesize)
    fanwei[0]=score[0]
    #计算每个点的得分
    for i in range(1,thesize):
        fanwei[i]=fanwei[i-1]+score[i]

    choose = [-1,-1,-1,-1]
    first = random.random()
    for i in range(0,thesize):
        if first<fanwei[i]:
            choose[0] = i
            break
    while True:
        first = random.random()
        flag = 1
        for i in range(0, thesize):
            if first < fanwei[i]:
                choose[1] = i
                if choose[1] != choose[0]:
                    flag = 0
                break
        if flag == 0:
            break
    while True:
        first = random.random()
        flag = 1
        for i in range(0, thesize):
            if first < fanwei[i]:
                choose[2] = i
                if choose[2] != choose[0] and choose[2] != choose[1]:
                    flag = 0
                break
        if flag == 0:
            break
    while True:
        first = random.random()
        flag = 1
        for i in range(0, thesize):
            if first < fanwei[i]:
                choose[3] = i
                if choose[3] != choose[0] and choose[3] != choose[1]and choose[3] != choose[2]:
                    flag = 0
                break
        if flag == 0:
            break
    # print(choose)
    return choose


