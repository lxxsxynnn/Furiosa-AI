import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

# 데이터가 작아 실행할 때마다 결과가 크게 흔들리므로 시드 고정
np.random.seed(337)
tf.random.set_seed(337)

# 1. 데이터
# train data 수치화 준비
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    # width_shift_range=0.1,  # 평행이동
    # rotation_range=5,       # 각도 조절(입력한 값만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    # fill_mode='nearest',    # 데이터를 옮기면 빈 공간이 생기게 되는데 그 근처 값으로 채우겠다는 의미
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)   # 테스트 데이터는 스케일링만

path_train = 'C:/study/_data/image/brain/train/'
path_test = 'C:/study/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),     # 알아서 크기 조정 해줌
    batch_size=200,             # 제너레이터가 한 번에 꺼내주는 이미지 묶음의 개수
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백
    shuffle=True,
)

# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,             # 테스트 데이터는 셔플할 필요가 없음
)

# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)     # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)       # (120, 150, 150, 1) (120,)

# 2. 모델 구성
model = Sequential()
# BatchNormalization 제거 : 훈련 때는 배치 통계, 평가 때는 누적 평균을 쓰는데
# 훈련 데이터가 128장뿐이라 둘이 어긋나 val_acc가 0.375에 굳고 예측이 한쪽으로 쏠렸음
model.add(Conv2D(50, (2, 2), input_shape=(150, 150, 1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(Flatten())    # GAP으로 바꿔봤더니 0.575로 떨어짐 > 위치 정보가 필요한 데이터
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=100,     # val이 32장뿐이라 val_loss가 요동침
                   restore_best_weights=True,   # patience가 크면 초반 최저점으로 되돌아갈 수 있음
                  )

start = time.time()
model.fit(x_train, y_train, epochs=1000,
          batch_size=16,
          callbacks=[es],
          validation_split=0.2,
          verbose=1,        # val_acc가 0.5에서 안 움직이는지 확인
          )
end = time.time()

# 4. 평가, 예측
loss =model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

print(np.unique(y_pred, return_counts=True))    # 한 값만 나오면 한쪽으로만 찍고 있는 것

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           #
print('acc : ', round(loss[1], 4))                  #
print('acc_score : ', acc_score)                    #
print('time : ', round(end - start, 2), 'sec')      #
# 시드 337 고정, patience 30 으로 같은 조건에서 비교한 결과
# Flatten + BatchNormalization : acc 0.5000  (120장 전부 한쪽으로 예측)
# Flatten, BN 없음             : acc 0.9750  (63 / 57)
# GAP, BN 없음                 : acc 0.5750  (17 / 103)