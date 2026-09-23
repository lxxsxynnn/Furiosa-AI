import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ReduceLR - mnist 데이터셋
# 1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,  # 평행이동
    rotation_range=5,       # 각도 조절(입력한 값만큼 이미지 회전)
    fill_mode='nearest',    # 데이터를 옮기면 빈 공간이 생기게 되는데 그 근처 값으로 채우겠다는 의미
)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (48000, 28, 28, 1) (12000, 28, 28, 1)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 48000개 중에 40000개 랜덤뽑기
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

xy_augmented = data_gen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))

from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_val = ohe.transform(y_val)
y_test = ohe.transform(y_test)

print(y_train.shape, y_val.shape, y_test.shape)     # (88000, 10) (12000, 10) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2 ,2), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0099),
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=40,
                   restore_best_weights=True,
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=60,
    factor=0.4,
    verbose=1,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[es, rlr],
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.03474234417080879 > 0.03479604050517082
print('acc : ', loss[1])                                    # acc :  0.9905999898910522 > 0.9894000291824341

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score:  0.9906 > 0.9894
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  1680.93 sec > 1201.75 sec