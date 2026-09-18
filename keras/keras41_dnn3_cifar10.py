import time
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# DNN 모델로 이미지 처리 실습해보기 - cifar10 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 32 * 32 * 3)   # 컬러라 채널 3까지 곱함 -> (50000, 3072)
x_test = x_test.reshape(-1, 32 * 32 * 3)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape, y_test.shape)      # (50000, 100) (10000, 100)

model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Dense(1000, input_shape=(32 * 32 * 3,), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=25, activation='relu'))
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
    patience=50,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1,
          callbacks=[es],
          validation_split=0.2,
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 0.5716548562049866 > 1.276353359222412
print('acc : ', loss[1])                                    # acc : 0.8070999979972839 > 0.5508999824523926

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.8071 > 0.5509
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 136.29 sec > 69.56 sec