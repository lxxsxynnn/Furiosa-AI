from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG19(include_top=False, input_shape=(100, 100, 3)),
              ResNet50(include_top=False, input_shape=(100, 100, 3)),
              ResNet101(include_top=False, input_shape=(100, 100, 3)),
              DenseNet201(include_top=False, input_shape=(100, 100, 3)),
              InceptionV3(include_top=False, input_shape=(100, 100, 3)),        # man_woman, cat_dog 같은 큰 데이터에 활용
              InceptionResNetV2(include_top=False, input_shape=(100, 100, 3)),
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(100, 100, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 모델마다 똑같은 데이터를 쓰므로 for 문 밖에서 한 번만 불러옴
np_path = 'C:/study/_save/numpy/man-woman_npy/'

x_train = np.load(np_path + 'keras_04_x_train.npy')
y_train = np.load(np_path + 'keras_04_y_train.npy')
x_test = np.load(np_path + 'keras_04_x_test.npy')
y_test = np.load(np_path + 'keras_04_y_test.npy')
# npy는 저장할 때 이미 0~1로 나눠서 스케일링하지 않음

print(x_train.shape, y_train.shape) # (20375, 100, 100, 3) (20375,)
print(x_test.shape, y_test.shape)   # (6792, 100, 100, 3) (6792,)

# validation_split은 섞기 전 마지막 20%를 가져가서, 클래스 순서대로 저장된 npy면 검증셋이 한쪽 클래스로 쏠림
# 미리 섞어서 검증셋을 떼어냄
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=333, stratify=y_train,
)

for model_i in model_list:
    model_i.trainable = False       # 동결 O : 불러온 모델의 imagenet 가중치는 고정, 아래 Dense 만 훈련
    # model_i.trainable = True      # 동결 X : 불러온 모델까지 전부 훈련

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))     # 이진분류

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    )

    start_time = time.time()

    model.fit(x_train,
              y_train,
              epochs=9999999,       # 사실상 무한(EarlyStopping 이 멈출 때까지 훈련)
              batch_size=32,
              verbose=0,
              validation_data=(x_val, y_val),
              callbacks=[es]
    )

    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('==============================')
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1],
          '시간 :', round(end_time - start_time, 2), '초')
