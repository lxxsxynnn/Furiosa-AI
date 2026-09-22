import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 데이터 증폭하기 - cifar10 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,   # 실사진이라 뒤집힌 자동차·개는 없음. 없는 상황을 학습시키게 됨
    zoom_range=0.2,
    fill_mode='nearest',
)

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# 검증셋을 원본에서 먼저 떼어낸다
# validation_split은 섞기 전 마지막 20%를 가져가는데, 증폭본을 뒤에 붙이면 검증셋이 전부 증폭본이 된다
# 테스트가 원본이므로 검증도 원본이어야 하고, 증폭은 훈련셋에만 적용한다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (40000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 25000
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 40000개 중에 25000개 랜덤뽑기

x_augmented = x_train[randidx].copy()    # (25000, 32, 32, 3)
y_augmented = y_train[randidx].copy()    # (25000, 1)

# 컬러라 이미 4차원이므로 reshape이 필요 없음

xy_augmented = data_gen.flow(   # rescale=1./255가 여기서 걸려 0~1이 된다
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

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

print(y_train.shape, y_val.shape, y_test.shape)     # (65000, 10) (10000, 10) (10000, 10)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())     # Dropout보다 앞에. 꺼진 상태의 통계를 배우면 예측 때와 안 맞음
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
)

path = 'C:/study/_save/keras51/'
os.makedirs(path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras51_03_cifar10.keras'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=128,
          verbose=1,
          callbacks=[es, mcp],
          validation_data=(x_val, y_val),    # 원본에서 떼어낸 검증셋
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 0.5716548562049866 > 0.6933213472366333
print('acc : ', loss[1])                                    # acc : 0.8070999979972839 > 0.8224999904632568

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.8071 > 0.8225
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 136.29 sec > 309.94 sec