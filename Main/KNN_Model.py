import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

#thong so
img_sise = 150
dataset_dir = "dataset"

x = []
y = []

#load du lieu
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # doc anh
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_sise, img_sise))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_flatten = img.flatten()/250.0
        x.append(img_flatten)
        y.append(person_name)

#chuyen ve numpy
x = np.array(x)
y = np.array(y)

# traning model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

#luu model
with open('Model/face_reconition_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Đã huấn luyện và lưu model nhận diện khuôn mặt (KNN)")