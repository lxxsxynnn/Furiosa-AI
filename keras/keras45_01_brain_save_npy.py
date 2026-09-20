import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지를 numpy 배열로 바꿔 저장만 한다 (모델 훈련은 keras45_02에서)
# 매번 JPEG를 열고 리사이즈하는 시간을 한 번만 치르고, 그 결과를 파일로 남겨두는 것

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)   # 테스트 데이터는 스케일링만

path_train = 'C:/study/_data/image/brain/train/'
path_test = 'C:/study/_data/image/brain/test/'

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),     # 알아서 크기 조정 해줌
    batch_size=200,             # 160장보다 크므로 배치가 1개 > [0]이 곧 전체
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백
    shuffle=False,              # 저장본은 파일 순서 그대로 > 돌릴 때마다 같은 npy가 나옴
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
)
# Found 120 images belonging to 2 classes.

# flow_from_directory까지는 파일 목록만 세고, 실제로 JPEG를 열어
# 150x150으로 줄이고 배열로 바꾸는 건 [0]을 꺼내는 이 시점
x_train, y_train = xy_train[0]
x_test, y_test = xy_test[0]

end1 = time.time()

print(x_train.shape, y_train.shape)     # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)       # (120, 150, 150, 1) (120,)

# shuffle=False라서 앞 80장이 ad(0), 뒤 80장이 normal(1)로 정렬되어 있음
# 이대로 validation_split을 쓰면 검증셋이 전부 normal이 되므로 keras45_02에서 섞어 나눈다
print(np.unique(y_train, return_counts=True))   # (array([0., 1.]), array([80, 80]))

# 2. npy로 저장
np_path = 'C:/study/_data/kaggle_cat_dog_npy/'
os.makedirs(np_path, exist_ok=True)

start2 = time.time()

np.save(np_path + 'keras_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras_01_y_test.npy', arr=y_test)

end2 = time.time()

print('이미지 > numpy : ', round(end1 - start1, 2), 'sec')   # 0.12 sec
print('npy 저장       : ', round(end2 - start2, 2), 'sec')   # 0.01 sec

# keras45_02에서 npy를 불러오는 시간과 비교
