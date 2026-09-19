import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train data 수치화 준비
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동
    rotation_range=5,       # 각도 조절(입력한 값만큼 이미지 회전)
    zoom_range=1.2,
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 데이터를 옮기면 빈 공간이 생기게 되는데 그 근처 값으로 채우겠다는 의미
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)   # 테스트 데이터는 스케일링만

path_train = 'C:/study/_data/image/train/'
path_test = 'C:/study/_data/image/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),     # 알아서 크기 조정 해줌
    batch_size=10,              # 제너레이터가 한 번에 꺼내주는 이미지 묶음의 개수
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백
    shuffle=True,
)

# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(   # test_datagen(...)으로 호출하면 TypeError: not callable
    path_test,                               # test_datagen은 설정을 담은 객체이지 함수가 아니므로 메서드를 붙여야 함
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,             # 테스트 데이터는 셔플할 필요가 없음
)

# Found 120 images belonging to 2 classes.

print(xy_train)         # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(xy_train.next())  # 가장 처음 값을 꺼내옴
# print(xy_train.next())  # 두 번째도 x와 y가 모여있는 데이터가 출력됨
# Iterator를 한 번에 다 출력해서 보려면 > for문 사용

# print(xy_train[0])      # Iterator 첫 번째와 동일
# print(xy_train[1])      # Iterator 두 번째와 동일

# print(xy_train[0][0])   # 첫 번째 배치의 x data
# print(xy_train[0][1])   # 첫 번째 배치의 y data

print(xy_train[0][0].shape) # (10, 100, 100, 1) > batch_size 때문에 이렇게 나옴
print(xy_train[0][1].shape) # (10,)

# 제대로 하려면 whole batch 사용해야 함

# print(xy_train[8][0])
# print(xy_train[15][0])
# print(xy_train[100][0])     # error > 배치는 10개니까 160장이 나오기 때문에

print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> 
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>