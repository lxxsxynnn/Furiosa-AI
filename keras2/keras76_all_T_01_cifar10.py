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

model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet201(include_top=False, input_shape=(32, 32, 3)),
            #   InceptionV3(), InceptionResNetV2(), # man_woman, cat_dog 같은 큰 데이터에 활용
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 모델마다 똑같은 데이터를 쓰므로 for 문 밖에서 한 번만 불러옴
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

for model_i in model_list:
    model_i.trainable = False       # 동결 O : 불러온 모델의 imagenet 가중치는 고정, 아래 Dense 만 훈련
    # model_i.trainable = True      # 동결 X : 불러온 모델까지 전부 훈련

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

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

# ==============================
# 모델명 : vgg19 loss : 1.344775676727295 acc : 0.476500004529953
# ==============================
# 모델명 : resnet50 loss : 4.328162670135498 acc : 0.20090000331401825
# ==============================
# 모델명 : resnet101 loss : 2.5120909214019775 acc : 0.3361000120639801
# ==============================
# 모델명 : densenet201 loss : 1.0415445566177368 acc : 0.6588000059127808
# ==============================
# 모델명 : mobilenetv2_1.00_224 loss : 1.8287359476089478 acc : 0.3926999866962433