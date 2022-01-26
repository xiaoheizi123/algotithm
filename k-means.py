import numpy as np
import  matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math


def CreateDataset():
    return [[3,4],[1,2],[2,1],[11,11],[9,11],[5,4],[1,9],[7,3],[8,9],[10,7]]

def calDis(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def updateCenterPoints(points, center_points, cluster, changed):
    ori_center_points = center_points.copy()
    tmp_points = np.tile(points, (len(center_points),1)).reshape(2,10,-1)
    res = np.zeros((len(center_points), len(tmp_points[0])))
    for i in range(len(center_points)):
        res[i] = [calDis(x, center_points[i]) for x in tmp_points[i]]

    center_index = np.argmin(res, axis=0)
    cluster = []
    changed = 0
    for i in range(len(center_points)):
        cluster.append(np.asarray(points)[center_index==i])
        center_points[i] = [np.asarray(cluster[i])[:,0].sum()/len(cluster[i]), np.asarray(cluster[i])[:,1].sum()/len(cluster[i])]
        changed += calDis(ori_center_points[i], center_points[i])
    return center_points, cluster, changed

def Kmeans(points, k, verbose):
    center_points = random.sample(points, k)

    changed = 1000
    cluster = []
    while(changed > 2):
        center_points, cluster, changed = updateCenterPoints(points, center_points, cluster, changed)

    if verbose:
        # 画出点和点群中心点
        color = ['green', 'blue', 'yellow', 'black', 'brown', 'pink', 'purple', 'yellowgreen']
        for i in range(len(cluster)):
            for j in range(len(cluster[i])):
                plt.scatter(cluster[i][j][0], cluster[i][j][1], marker='o', color=color[i], s=30, )
        for i in range(len(center_points)):
            plt.scatter(center_points[i][0], center_points[i][1], marker='o', color='red', s=30, )
        plt.show()

    return center_points, cluster

if __name__ == "__main__":
    point_dataset = CreateDataset()     # 待聚类的坐标点
    k = 2                               # 聚类的个数
    verbose = True                      # 聚类结果可视化
    center_points, cluster = Kmeans(point_dataset, k, verbose)
    # print(center_points)
