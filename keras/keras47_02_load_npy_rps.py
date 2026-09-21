import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

# 1. 데이터
np_path = 'C:/study/_save/numpy/rps_npy/'

x_train = np.load(np_path + 'keras_02_x_train.npy')
y_train = np.load(np_path + 'keras_02_y_train.npy')
x_test = np.load(np_path + 'keras_02_x_test.npy')
y_test = np.load(np_path + 'keras_02_y_test.npy')

print(x_train.shape, y_train.shape) # (1536, 100, 100, 3) (1536, 3)
print(x_test.shape, y_test.shape)   # (512, 100, 100, 3) (512, 3)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Conv2D(100, (2, 2), activation='relu'))
model.add(Conv2D(100, (2, 2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(100, (2, 2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), activation='relu'))
model.add(Conv2D(25, (2, 2), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))   # 클래스 3개 > 노드 3개

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=10,
                   )

model.fit(x_train, y_train, epochs=100,
          batch_size=32,
          callbacks=[es],
          validation_split=0.2,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)           # (512, 3) 확률 3개씩
y_predict = np.argmax(y_predict, axis=1)    # 가장 큰 쪽의 번호 > 0/1/2
y_test = np.argmax(y_test, axis=1)          # y_test도 원핫이라 같이 되돌린다

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])                           # loss :  0.00011715513392118737
print('acc : ', round(loss[1], 4))                  # acc :  1.0
print('acc_score : ', acc_score)                    # acc_score :  1.0