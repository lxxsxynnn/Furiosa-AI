import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 개/고양이 이미지를 numpy 배열로 바꿔 저장만 한다 (모델 훈련은 keras45_04에서)
# brain(280장)과 달리 8005장이라 JPEG를 여는 시간이 실제로 체감된다

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)   # 테스트 데이터는 스케일링만

path_train = 'C:/study/_data/image/catdog/training_set/'
path_test = 'C:/study/_data/image/catdog/test_set/'

SIZE = (100, 100)   # 200x200이면 npy가 4.8GB. 100x100이면 1.2GB이고 훈련도 4배 빠름
BATCH = 100         # 여기서는 꺼내는 단위일 뿐 (훈련 배치 크기가 아님)

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode='binary',        # 이진분류
    color_mode='rgb',           # 컬러 > 채널 3
    shuffle=False,              # 저장본은 파일 순서 그대로 > 돌릴 때마다 같은 npy
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)
# Found 2023 images belonging to 2 classes.

print(xy_train.class_indices)       # {'cats': 0, 'dogs': 1}


def to_numpy(gen):
    """배치를 전부 순회해 하나의 numpy 배열로 합친다.

    xy[0][0]만 꺼내면 첫 배치(BATCH장)뿐이므로 전체를 담으려면 모든 배치를 돌아야 한다.
    concatenate는 중간 리스트 때문에 메모리를 두 배로 쓰므로 미리 잡아두고 채워 넣는다.
    """
    h, w = gen.target_size
    c = 3 if gen.color_mode == 'rgb' else 1
    x = np.empty((gen.samples, h, w, c), dtype=np.float32)
    y = np.empty((gen.samples,), dtype=np.float32)
    i = 0
    for b in range(len(gen)):       # len(gen) = 배치 개수
        bx, by = gen[b]
        x[i:i+len(bx)] = bx
        y[i:i+len(by)] = by
        i += len(bx)
    return x, y


x_train, y_train = to_numpy(xy_train)
x_test, y_test = to_numpy(xy_test)

end1 = time.time()

print(x_train.shape, y_train.shape)     # (8005, 100, 100, 3) (8005,)
print(x_test.shape, y_test.shape)       # (2023, 100, 100, 3) (2023,)
print(round(x_train.nbytes / 1024**3, 2), 'GB')     # 0.89 GB

# shuffle=False라서 앞쪽이 cats(0), 뒤쪽이 dogs(1)로 정렬되어 있음
# 이대로 validation_split을 쓰면 검증셋이 전부 dogs가 되므로 keras45_04에서 섞어 나눈다
print(np.unique(y_train, return_counts=True))   # (array([0., 1.]), array([4000, 4005]))

# 2. npy로 저장
np_path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'
os.makedirs(np_path, exist_ok=True)

start2 = time.time()

np.save(np_path + 'keras45_03_x_train.npy', arr=x_train)
np.save(np_path + 'keras45_03_y_train.npy', arr=y_train)
np.save(np_path + 'keras45_03_x_test.npy', arr=x_test)
np.save(np_path + 'keras45_03_y_test.npy', arr=y_test)

end2 = time.time()

print('이미지 > numpy : ', round(end1 - start1, 2), 'sec')   # 6.76 sec
print('npy 저장       : ', round(end2 - start2, 2), 'sec')   # 0.92 sec

# keras45_04에서 npy를 불러오는 시간과 비교
