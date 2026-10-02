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

# 리셰이프 레이어의 위치
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

print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_test.shape)   # (60000,) (10000,)

# 2. 모델 구성
from tensorflow.keras.layers import Reshape, LSTM

model = Sequential()
model.add(Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(64, (3 ,3)))  # (26, 26, 64)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))    # (24, 24, 32)
model.add(Conv2D(16, (2, 2), activation='relu'))    # (23, 23, 16)
##########################################################
model.add(Reshape(target_shape=(23 * 23, 16)))
model.add(LSTM(10))
##########################################################
model.add(Dense(units=32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0099),
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=10,
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
model.fit(x_train, y_train, epochs=20, batch_size=128,
          verbose=1,
          callbacks=[es, rlr],
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.03479604050517082
print('acc : ', loss[1])                                    # acc :  0.9894000291824341

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score:  0.9894
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  1201.75 sec