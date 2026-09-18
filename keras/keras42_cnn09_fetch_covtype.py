from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import numpy as np
import pandas as pd

# 2차원 데이터를 CNN 모델로 처리해보기 - 산림 피복 유형
# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)                 # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    stratify=y,
    random_state=12,
)

print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)
print(y_train.shape, y_test.shape)  # (464809, 7) (116203, 7)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 9, 6, 1)
x_test = x_test.reshape(-1, 9, 6, 1)

print(x_train.shape, x_test.shape)  # 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(9, 6, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'],
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
model.fit(x_train, y_train,
          epochs=200,
          batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es,],
          )

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])
print('acc : ', round(result[1], 2))
print('acc_score : ', acc_score)

'''
loss :  0.38224494457244873
acc :  0.85
acc_score :  0.8505374215811984

>>

loss :  0.2234436720609665
acc :  0.91
acc_score :  0.9102690980439403
'''