import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# GlobalAveragePooling 실습해보기 - mnist 데이터셋
# 1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

# 3차원 데이터로 변환하기
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# reshape 이유 : OneHotEncoder는 2차원 입력만 받음 (위 ValueError)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2 ,2), input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=40,
                   restore_best_weights=True,
                   )

start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.025623517110943794 > 0.03741923347115517
print('acc : ', loss[1])                                    # acc :  0.9947999715805054 > 0.991599977016449

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score: 0.9948 > 0.9916
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 399.21 sec > 940.37 sec