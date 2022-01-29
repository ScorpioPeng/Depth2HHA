import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame
from pyntcloud.utils.array import PCA


def main():
    points = np.genfromtxt("1_point.txt", delimiter=",")
    points = DataFrame(points[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化

    data_mean = np.mean(points, axis=0)  #对列求取平均值
    # 归一化
    normalize_data = points - data_mean
    # SVD分解
    # 构造协方差矩阵
    H = np.dot(normalize_data.T, normalize_data)
    # SVD分解
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)   # H = U S V
    # 逆序排列
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]

    print(eigenvectors[:, 0])  #最大特征值 对应的特征向量，即第一主成分
    print(eigenvectors[:, 1])  #第二主成分
    '''
    #在原点云中画图
    point = [[0, 0, 0], eigenvectors[:, 0], eigenvectors[:, 1]]  # 提取第一v和第二主成分 也就是特征值最大的对应的两个特征向量 第一个点为原点
    lines = [[0, 1], [0, 2]]  # 有点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和两个成分向量划线
    colors = [[1, 0, 0], [0, 1, 0]]  # 为不同的线添加不同的颜色
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])
    '''
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)  # 对点云建立kd树 方便搜索
    normals = []
    # print(point_cloud_o3d)  #geometry::PointCloud with 10000 points.
    print(points.shape[0])  # 10000
    for i in range(points.shape[0]):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)  # 10 个临近点
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])
    for i in normals:
        f = open('normal.txt', 'a')
        f.write( str(i).strip('[').strip(']')+'\n')
        f.close()
if __name__ == '__main__':
    main()