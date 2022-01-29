'''
    生成人脸数据和相应的脸部点云数据（训练集）
'''
from tools.Face_detection import segmentation_face
import os
import cv2
import numpy as np
from tools.getHHA import getHHA_img

root = "data/traindata"
train_data = "training_data"
train_hha = 'training_hha'

mini_dataset = []  # 原图子数据集
for i in range(1, 7):
    name = 'CCF10' + str(i).rjust(2, '0') + 'gaze'
    mini_dataset.append(name)


# 创建子文件夹traindata\traning_cloud2500\CCF1001gaze等
# 创建子文件夹traindata\train_face\CCF1001gaze等
for name in mini_dataset:
    dstpath1 = os.path.join(root, train_hha, name)  # 创建路径
    if not os.path.exists(dstpath1):
        os.makedirs(dstpath1)


for nth in mini_dataset:  # 每个子文件夹

    sub_dataset = os.path.join(root, train_data, nth)  # RGB、depth图子数据集 traindata\training_data\CCF1001gaze
    sub_HHAset = os.path.join(root,train_hha, nth)    # HHA数据集 traindata\training_hha\CCF1001gaze
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
    C = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  #相机矩阵

    for roots, dirs, files in os.walk(sub_dataset):#遍历数据集
        num_image = int((len(files)-1) / 3)   #获取每个子数据集中的样本数量
        for index in range(num_image):
            index += 1
            # 读取相应的彩色图像和深度图像，并进行人脸框选
            img = os.path.join(sub_dataset, str(index) + '_color.jpg')
            depth_img = os.path.join(sub_dataset, str(index) + '_depth.png')
            hha_name = os.path.join(sub_HHAset, str(index) + '_hha.png')
            print("正在处理：",img)
            # depth_scale, rgb_scale, top, left = segmentation_face(img, depth_img)
            # D = cv2.resize(depth_scale,(448,448))
            dimg = cv2.imread(depth_img, -1)/1000
            hha_complete = getHHA_img(C, dimg, dimg)
            cv2.imwrite(hha_name, hha_complete )
