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
            #   InceptionV3(), InceptionResNetV2(), # man_woman, cat_dog 같은 큰 데이터에 활용
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(100, 100, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 모델마다 똑같은 데이터를 쓰므로 for 문 밖에서 한 번만 불러옴
np_path = 'C:/study/_save/numpy/rps_npy/'

x_train = np.load(np_path + 'keras_02_x_train.npy')
y_train = np.load(np_path + 'keras_02_y_train.npy')
x_test = np.load(np_path + 'keras_02_x_test.npy')
y_test = np.load(np_path + 'keras_02_y_test.npy')
# npy는 저장할 때 이미 0~1로 나눠서 스케일링하지 않음

print(x_train.shape, y_train.shape) # (1536, 100, 100, 3) (1536, 3)
print(x_test.shape, y_test.shape)   # (512, 100, 100, 3) (512, 3)

for model_i in model_list:
    model_i.trainable = False       # 동결 O : 불러온 모델의 imagenet 가중치는 고정, 아래 Dense 만 훈련
    # model_i.trainable = True      # 동결 X : 불러온 모델까지 전부 훈련

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(3, activation='softmax'))     # 클래스 3개, y가 원핫이라 categorical_crossentropy

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
              validation_split = 0.2,
              callbacks=[es]
    )

    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('==============================')
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1],
          '시간 :', round(end_time - start_time, 2), '초')
