import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
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

'''
ValueError: Expected 2D array, got 1D array instead:
array=[5 0 4 ... 5 6 8].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
'''

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
# model.add(Conv2D(64, (3 ,3), input_shape=(28, 28, 1))) # (26, 26, 64)
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')) # (24, 24, 32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2, 2), activation='relu')) # (23, 23, 32)
# model.add(Conv2D(16, (2, 2), activation='relu')) # (22, 22, 16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2, 2), activation='relu')) # (21, 21, 16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2, 2), activation='relu')) # (20, 20, 16) > 3차원 상태로는 softmax 적용이 힘듦
# model.add(Flatten())
# model.add(Dense(units=32, activation='relu'))    # 값을 모아줘야 함
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape=(32, ), activation='relu'))
# model.add(Dense(10, activation='softmax'))       # (10,)
# model.summary()

'''
Model: "sequential"
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640       
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
 dropout (Dropout)           (None, 24, 24, 32)        0         
 conv2d_2 (Conv2D)           (None, 23, 23, 32)        4128      
 conv2d_3 (Conv2D)           (None, 22, 22, 16)        2064      
 dropout_1 (Dropout)         (None, 22, 22, 16)        0         
 conv2d_4 (Conv2D)           (None, 21, 21, 16)        1040      
 dropout_2 (Dropout)         (None, 21, 21, 16)        0         
 conv2d_5 (Conv2D)           (None, 20, 20, 16)        1040      
 flatten (Flatten)           (None, 6400)              0         
 dense (Dense)               (None, 32)                204832 > w = 6400, b = 32    
 dropout_3 (Dropout)         (None, 32)                0         
 dense_1 (Dense)             (None, 16)                528       > 실제 연산이 이루어지는 곳
 dense_2 (Dense)             (None, 10)                170                                                                        
=================================================================
Total params: 232,906
Trainable params: 232,906
Non-trainable params: 0
'''

model.add(Conv2D(64, (2 ,2), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(Flatten())
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
print('================ model.evaluate ================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.042573001235723495
print('acc : ', loss[1])                                    # acc :  0.9898999929428101

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score :  0.9899
# predict 이유 : evaluate는 케라스가 내부에서 예측까지 해주지만,
#                accuracy_score는 sklearn 함수라 예측값 배열을 직접 넘겨야 함
#                확률·원핫이라 둘 다 argmax로 라벨을 되돌린 뒤 비교
#                evaluate의 acc와 값이 같으면 argmax 처리에 실수가 없다는 뜻
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  610.07 sec (CPU) / 141.17 sec (GPU)

# 0.995 맞추기
