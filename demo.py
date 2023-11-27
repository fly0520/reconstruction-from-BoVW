import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import vq

# Load Caltech101 dataset
# dataset = load_caltech101_dataset()  # Implement this function to load your dataset

# Parameters
image_size = (128, 128)
n_clusters = 5000
patch_size = 32
step_size = 8

# Function to extract SIFT features from an image
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints = [cv2.KeyPoint(x, y, patch_size) for y in range(0, image.shape[0], step_size)
                                            for x in range(0, image.shape[1], step_size)]
    _, descriptors = sift.compute(image, keypoints)
    return descriptors

# Resize images and extract features
all_features = []
for img in dataset:
    img_resized = cv2.resize(img, image_size)
    descriptors = extract_sift_features(img_resized)
    if descriptors is not None:
        all_features.extend(descriptors)

# K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(all_features)

# Create BoVW for each image
bovw_features = []
for img in dataset:
    img_resized = cv2.resize(img, image_size)
    descriptors = extract_sift_features(img_resized)
    if descriptors is not None:
        words, _ = vq(descriptors, kmeans.cluster_centers_)
        img_bovw = np.zeros(n_clusters)
        for w in words:
            img_bovw[w] += 1
        bovw_features.append(img_bovw)

# Standardize features
scaler = StandardScaler()
bovw_features = scaler.fit_transform(bovw_features)


def reconstruct_images(model, loader, num_images=5):
    for i, (data, _) in enumerate(loader):
        recon, _, _ = model(data)
        if i == 0:
            comparison = torch.cat([data[:num_images], recon.view(64, 1, 128, 128)[:num_images]])
            save_image(comparison.cpu(), 'reconstruction.png', nrow=num_images)
            break

# 加载模型并重建图像
load_model(model)
reconstruct_images(model, train_loader)