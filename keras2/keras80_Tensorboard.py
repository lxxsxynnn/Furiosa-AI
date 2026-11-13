import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

# cnn 실습해보기 - mnist 데이터셋
# 1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

######## 스케일링 1 ########
# x_train = x_train / 255.    # 이미지 데이터라 255로 나눔
# x_test = x_test / 255.      # float 형태로 사용하기 위에 뒤에 .을 붙였음

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

######## 스케일링 2 ######## (이미지에서 많이 사용하는 방법)
x_train = (x_train - 127.5) / 127.5   # 범위를 -1 ~ 1 사이로 만들어주기 위해서 255의 절반인 127.5를 빠고 127.5로 나눔
x_test = (x_test - 127.5) / 127.5     # MinMax와 다르게 범위 밖으로 값이 나갈 일이 없어서 값을 따로 떼놓을 필요 없음

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

# 3차원 데이터로 변환하기
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)
# reshape 이유 : Conv2D의 input_shape는 (세로, 가로, 채널) 3차원인데
#                MNIST는 흑백이라 채널 축 없이 (60000, 28, 28)로 불러와짐 -> 채널 1을 붙임

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = ohe.fit_transform(y_train)

# y_train = y_train.reshape(60000, 1) # 전체 데이터 수를 알 경우
y_train = y_train.reshape(-1, 1)    # -1을 넣어도 전체 데이터가 됨
y_test = y_test.reshape(-1, 1)
# reshape 이유 : OneHotEncoder는 2차원 입력만 받음 (위 ValueError)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)
# 원핫 이유 : 1) 라벨 0~9는 이름일 뿐인데 정수로 두면 "9는 1보다 크다"로 학습됨
#             2) 출력층이 Dense(10, softmax)라 예측이 (10,) -> 정답도 (10,)이어야 비교됨

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3 ,3), input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

tb = TensorBoard(log_dir="C:/study/_save/tensorboard/_keras80/",
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True,
                 )

start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, tb],
          )
end_time = time.time()

# 4. 평가, 예측
print('================ model.evaluate ================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.042573001235723495
print('acc : ', loss[1])                                    # acc :  0.9898999929428101

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score :  0.9899
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  610.07 sec (CPU) / 141.17 sec (GPU)

'''
cmd 창에서 경로 따라 들어간 뒤
tensorboard --logdir=.

경로 안맞추고 실행 시
tensorboard --logdir=C:/study/tensorboard/_keras80
'''