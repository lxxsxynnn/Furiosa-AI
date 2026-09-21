import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
path = 'C:/study/_data/image/rps/'

datagen = ImageDataGenerator(
    rescale = 1./255,
)

xy = datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=2100,            # 전체 2049장보다 크게. 작으면 뒤쪽 클래스가 통째로 빠진다
    class_mode='categorical',   # 클래스 3개 > 원핫
    color_mode='rgb',
    shuffle=False,
)

x, y = xy[0]
print(x.shape, y.shape)         # (2049, 100, 100, 3) (2049, 3)

train_x, test_x, train_y, test_y = train_test_split(
    x, y,
    train_size=0.75,
    shuffle=True,
    random_state=243,
    stratify=y.argmax(axis=1),  # 원핫이라 라벨 번호로 되돌려서 넘긴다
)

np_path = 'C:/study/_save/numpy/rps_npy/'
os.makedirs(np_path, exist_ok=True)

np.save(np_path + 'keras_02_x_train.npy', arr=train_x)
np.save(np_path + 'keras_02_y_train.npy', arr=train_y)
np.save(np_path + 'keras_02_x_test.npy', arr=test_x)
np.save(np_path + 'keras_02_y_test.npy', arr=test_y)