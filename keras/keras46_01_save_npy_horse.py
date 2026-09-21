import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
path = 'C:/study/_data/image/horse-or-human/'

datagen = ImageDataGenerator(
    rescale = 1./255,
)

xy = datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=1100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

x, y = xy[0]

train_x, test_x, train_y, test_y = train_test_split(
    x, y,
    train_size=0.75,
    shuffle=True,
    random_state=243,
    stratify=y,
)

np_path = 'C:/study/_save/numpy/horse_npy/'
os.makedirs(np_path, exist_ok=True)

np.save(np_path + 'keras_01_x_train.npy', arr=train_x)
np.save(np_path + 'keras_01_y_train.npy', arr=train_y)
np.save(np_path + 'keras_01_x_test.npy', arr=test_x)
np.save(np_path + 'keras_01_y_test.npy', arr=test_y)