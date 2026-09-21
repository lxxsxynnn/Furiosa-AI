# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
path = 'C:/study/_data/image/man-woman/'

datagen = ImageDataGenerator(
    rescale = 1./255,
)

xy = datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=27200,           # 전체 27167장보다 크게. 작으면 뒤쪽 클래스가 통째로 빠진다
    class_mode='binary',        # man, woman 2개
    color_mode='rgb',
    shuffle=False,
)

x, y = xy[0]
print(x.shape, y.shape)         # (27167, 100, 100, 3) (27167,)

train_x, test_x, train_y, test_y = train_test_split(
    x, y,
    train_size=0.75,
    shuffle=True,
    random_state=243,
    stratify=y,                 # man 17678 : woman 9489로 기울어 있어 비율 유지
)

np_path = 'C:/study/_save/numpy/man-woman_npy/'
os.makedirs(np_path, exist_ok=True)

np.save(np_path + 'keras_04_x_train.npy', arr=train_x)
np.save(np_path + 'keras_04_y_train.npy', arr=train_y)
np.save(np_path + 'keras_04_x_test.npy', arr=test_x)
np.save(np_path + 'keras_04_y_test.npy', arr=test_y)