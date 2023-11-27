import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from os import listdir
from os.path import isfile, join
import random


# 参数设置
n_clusters = 5000  # 字典大小
patch_size = 32  # 图像块大小
step_size = 8  # 提取步长
image_size = (128, 128)  # 图像大小
sampling_fraction = 0.1  # 采样比例

# SIFT特征提取函数
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints = [cv2.KeyPoint(x, y, patch_size) for y in range(0, image.shape[0], step_size) 
                                            for x in range(0, image.shape[1], step_size)]
    _, descriptors = sift.compute(image, keypoints)
    return descriptors

# 从Caltech数据集中提取特征
all_features = []
base_path = "data/train"  

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        all_images = os.listdir(folder_path)
        sampled_images = random.sample(all_images, int(len(all_images) * sampling_fraction))
        for img_file in sampled_images:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, image_size)
            descriptors = extract_sift_features(img_resized)
            if descriptors is not None:
                all_features.extend(descriptors)

# K-means聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(all_features)

# 保存字典中心
dictionary = kmeans.cluster_centers_
np.save('visual_dictionary.npy', dictionary)
