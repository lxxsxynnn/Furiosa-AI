import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(337)
tf.random.set_seed(337)

# keras45_03이 저장해 둔 npy를 불러온다
# ImageDataGenerator도 flow_from_directory도 없음 > 이미지 폴더를 아예 건드리지 않는다

# 1. 데이터
np_path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'

start1 = time.time()

x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

end1 = time.time()

print(x_train.shape, y_train.shape)     # (8005, 100, 100, 3) (8005,)
print(x_test.shape, y_test.shape)       # (2023, 100, 100, 3) (2023,)
print('npy 로드 : ', round(end1 - start1, 2), 'sec')    # 0.36 sec

# npy는 저장된 순서 그대로라 cats 4000장 > dogs 4005장으로 정렬되어 있음
# validation_split은 뒤에서부터 잘라가므로 그냥 쓰면 검증셋이 전부 dogs가 됨
# 제너레이터는 매 에폭 알아서 섞어주지만 numpy는 내가 섞어야 한다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=453437,
    stratify=y_train,       # 검증셋에도 0과 1이 반반 들어가도록
)

print(x_train.shape, x_val.shape)               # (6404, 100, 100, 3) (1601, 100, 100, 3)
print(np.unique(y_val, return_counts=True))     # (array([0., 1.]), array([800, 801]))

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(50, (2, 2), input_shape=(100, 100, 3), padding='same'))    # 컬러라 채널 3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=5,
                   restore_best_weights=True,
                  )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=np_path + 'keras45_04_catdog.keras'
)

start2 = time.time()
model.fit(x_train, y_train, epochs=30,
          batch_size=32,
          callbacks=[es, mcp],
          validation_data=(x_val, y_val),   # 위에서 stratify로 갈라둔 검증셋
          verbose=1,
          )
end2 = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

print(np.unique(y_pred, return_counts=True))    # 한 값만 나오면 한쪽으로만 찍고 있는 것

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # 0.4994
print('acc : ', round(loss[1], 4))                  # 0.7578
print('acc_score : ', acc_score)                    # 0.7577854671280276
print('npy 로드 : ', round(end1 - start1, 2), 'sec')    # 0.36 sec
print('훈련     : ', round(end2 - start2, 2), 'sec')    # 406.31 sec

# 데이터 준비 시간 비교 (catdog 10,028장, 100x100 컬러)
# keras45_03  이미지 > numpy : 6.76 sec
# keras45_04  npy 로드       : 0.36 sec     > 약 19배
#
# brain(280장)은 0.12 sec vs 0.03 sec 라 차이가 안 느껴졌지만
# 장수가 늘면 JPEG를 열고 디코딩하고 리사이즈하는 비용이 그대로 커진다.
# npy는 이미 만들어진 배열을 그대로 읽으므로 그 과정이 통째로 없다.

# 예측 분포 (array([0., 1.]), array([1085, 938])) > 한쪽으로 쏠리지 않음
