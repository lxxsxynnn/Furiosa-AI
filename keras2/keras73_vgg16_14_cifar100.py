import random
import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import os
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    shear_range=20,
    fill_mode='nearest',
)

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# 검증셋을 원본에서 먼저 떼어낸다
# validation_split은 섞기 전 마지막 20%를 가져가는데, 증폭본을 뒤에 붙이면 검증셋이 전부 증폭본이 된다
# 테스트가 원본이므로 검증도 원본이어야 하고, 증폭은 훈련셋에만 적용한다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (40000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 25000
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 40000개 중에 25000개 랜덤뽑기

x_augmented = x_train[randidx].copy()    # (25000, 32, 32, 3)
y_augmented = y_train[randidx].copy()    # (25000, 1)

# 컬러라 이미 4차원이므로 reshape이 필요 없음

xy_augmented = next(data_gen.flow(   # rescale=1./255가 여기서 걸려 0~1이 된다
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
))  # Keras 3에는 .next() 메서드가 없어서 next()로 꺼냄

# 증폭본이 0~1이므로 나머지도 맞춘다
x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_val = ohe.transform(y_val)
y_test = ohe.transform(y_test)

# 2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)

vgg16.trainable=False

# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.00001),   # VGG16 가중치까지 학습하므로 lr을 작게. 크면 출력이 0으로 굳어 acc 고정
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
)

path = 'C:/study/_save/keras73/'
os.makedirs(path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras73_cifar100.keras'
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    factor=0.5,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=128,
          verbose=1,
          callbacks=[es, mcp, rlr],
          validation_data=(x_val, y_val),
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('time : ', round(end_time - start_time, 2), 'sec')


# 1. 이전에 만들었던 최상의 결과와 비교
# loss : 2.002082109451294
# acc : 0.4668999910354614
# accuracy_score : 0.4669
# time : 2205.06 sec

# 2. 가중치를 동결하고 훈련시켰을 때와 비교(trainable=False)
'''
loss :  2.859553575515747
acc :  0.29339998960494995
accuracy_score :  0.2934
time :  958.34 sec
'''

# 3. 가중치를 동결하지 않고 훈련시켰을 때와 비교(trainable=True)
'''
loss :  2.034325122833252
acc :  0.5088000297546387
accuracy_score :  0.5088
time :  515.19 sec
'''