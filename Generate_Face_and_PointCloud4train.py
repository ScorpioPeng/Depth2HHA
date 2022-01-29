'''
    生成人脸数据和相应的脸部点云数据（训练集）
'''
from tools.Face_detection import segmentation_face
import os
import cv2
import numpy as np
import torch
from tools.FPS import farthest_point_sample

root = "data/traindata"
train_data = "training_data"
train_cloud = "traning_cloud2500"
train_face='training_face'

mini_dataset = []  # 原图子数据集
for i in range(1, 25):
    name = 'CCF10' + str(i).rjust(2, '0') + 'gaze'
    mini_dataset.append(name)
for i in range(43, 61):
    name = 'CCF10' + str(i).rjust(2, '0') + 'gaze'
    mini_dataset.append(name)

# 创建子文件夹traindata\traning_cloud2500\CCF1001gaze等
# 创建子文件夹traindata\train_face\CCF1001gaze等
for name in mini_dataset:
    dstpath1 = os.path.join(root, train_face, name)  # 创建路径
    dstpath2 = os.path.join(root,train_cloud, name)  # 创建路径
    if not os.path.exists(dstpath1):
        os.makedirs(dstpath1)
    if not os.path.exists(dstpath2):
        os.makedirs(dstpath2)

for nth in mini_dataset:  # 每个子文件夹

    sub_dataset = os.path.join(root, train_data, nth)  # RGB、depth图子数据集 traindata\training_data\CCF1001gaze
    sub_faceset = os.path.join(root,train_face, nth)    # 框选人脸数据集 traindata\train_face\CCF1001gaze
    sub_cloudset = os.path.join(root, train_cloud, nth)  # 相应的人脸点云子数据集 traindata\traning_cloud2500\CCF1001gaze

    #获取相机参数
    camera_paras = os.path.join(sub_dataset, "camera_param.txt")
    f = open(camera_paras)  # 返回一个文件对象
    line = f.readlines()  # 调用文件的 readline()方法,读取相机参数
    paras = line[3].split()  # 分割
    cx = float(paras[0])  # optical center x
    cy = float(paras[1])  # optical center y
    fx = float(paras[2])  # focal length x
    fy = float(paras[3])  # focal length y
    factor = 1000
    f.close()

    for roots, dirs, files in os.walk(sub_dataset):#遍历数据集
        num_image = int((len(files)-1) / 3)   #获取每个子数据集中的样本数量
        for index in range(num_image):
            index += 1
            # 读取相应的彩色图像和深度图像，并进行人脸框选
            img = os.path.join(sub_dataset, str(index) + '_color.jpg')
            depth_img = os.path.join(sub_dataset, str(index) + '_depth.png')
            face_name = os.path.join(sub_faceset, str(index)+'_face.jpg')
            txt_name = os.path.join(sub_cloudset, str(index) + '_point.txt')
            print("正在处理：",img)
            depth_scale, rgb_scale, top, left = segmentation_face(img, depth_img)
            img448 = cv2.resize(rgb_scale, (448, 448))
            cv2.imwrite(face_name, img448)# 保存框选出的人脸数据到指定路径

            h, w, c = depth_scale.shape
            result_set = []
            for v in range(h):
                for u in range(w):
                    Z = depth_scale[v, u] / factor
                    X = (u + left - cx) * Z / fx
                    Y = (v + top - cy) * Z / fy
                    sumxyz = float(abs(X) + abs(Y) + abs(Z))
                    if sumxyz != 0:
                        resultxyz =[float(X),float(Y),float(Z)]
                        result_set.append(resultxyz)

            array = np.asarray([result_set])
            array2 = torch.from_numpy(array)
            array3 = array2.cuda()
            #采用farthest_point_sample算法，必须放到GPU上并行计算
            indexs=farthest_point_sample(array3,2500).cpu().numpy()[0]

            with open(txt_name,'w') as f:  #将FPS算法均匀挑选出的点全部写到样本的点云txt
                for kk in indexs:
                    f.write(str(result_set[kk]).strip('[').strip(']').replace(' ', '') + '\n')
            f.close()
