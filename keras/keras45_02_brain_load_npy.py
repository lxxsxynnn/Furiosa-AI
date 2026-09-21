import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(337)
tf.random.set_seed(337)

# keras45_01이 저장해 둔 npy를 불러온다
# ImageDataGenerator도 flow_from_directory도 없음 > 이미지 폴더를 아예 건드리지 않는다

# 1. 데이터
np_path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'

start1 = time.time()

x_train = np.load(np_path + 'keras_01_x_train.npy')
y_train = np.load(np_path + 'keras_01_y_train.npy')
x_test = np.load(np_path + 'keras_01_x_test.npy')
y_test = np.load(np_path + 'keras_01_y_test.npy')

end1 = time.time()

print(x_train.shape, y_train.shape)     # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)       # (120, 150, 150, 1) (120,)
print('npy 로드 : ', round(end1 - start1, 2), 'sec')    # 0.03 sec

# npy는 저장된 순서 그대로라 ad 80장 > normal 80장으로 정렬되어 있음
# validation_split은 뒤에서부터 잘라가므로 그냥 쓰면 검증셋이 전부 normal이 됨
# 제너레이터는 매 에폭 알아서 섞어주지만 numpy는 내가 섞어야 한다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=453437 ,
    stratify=y_train,       # 검증셋에도 0과 1이 반반 들어가도록
)

print(x_train.shape, x_val.shape)               # (128, 150, 150, 1) (32, 150, 150, 1)
print(np.unique(y_val, return_counts=True))     # (array([0., 1.]), array([16, 16]))

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(50, (2, 2), input_shape=(150, 150, 1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                  )

start2 = time.time()
model.fit(x_train, y_train, epochs=1000,
          batch_size=16,
          callbacks=[es],
          validation_data=(x_val, y_val),    # 위에서 stratify로 갈라둔 검증셋
          verbose=1,
          )
end2 = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

print(np.unique(y_pred, return_counts=True))    # 한 값만 나오면 한쪽으로만 찍고 있는 것

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # loss :  0.06783701479434967
print('acc : ', round(loss[1], 4))                  # acc :  0.9917
print('acc_score : ', acc_score)                    # 0.acc_score :  0.9916666666666667
print('npy 로드 : ', round(end1 - start1, 2), 'sec')    # 0.01 sec
print('훈련     : ', round(end2 - start2, 2), 'sec')    # 408.3 sec

# 데이터 준비 시간 비교 (brain 280장, 150x150 흑백)
# keras45_01  이미지 > numpy : 0.12 sec
# keras45_02  npy 로드       : 0.03 sec