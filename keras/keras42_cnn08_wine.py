from sklearn.datasets import load_wine
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# 2차원 데이터를 CNN 모델로 처리해보기 - 와인
# 1 . 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

print(x_train.shape, x_test.shape)  # 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(13, 1, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))     # 다중분류 출력층

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

save_path = 'C:/study/_save/keras33/'
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es,],
          )

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', result[0])
print('acc : ', round(result[1], 2))
print('acc_score : ', acc_score)

'''
loss :  0.20670649409294128
acc :  0.97
acc_score :  0.9722222222222222

>>

loss :  0.006112752482295036
acc :  1.0
acc_score :  1.0
'''