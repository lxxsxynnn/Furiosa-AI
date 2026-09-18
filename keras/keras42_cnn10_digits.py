from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import numpy as np

# 2차원 데이터를 CNN 모델로 처리해보기 - 손글씨 숫자
# 1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.reshape(-1 , 1))
print(y.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y,
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 8, 1)
x_test = x_test.reshape(-1, 8, 8, 1)

print(x_train.shape, x_test.shape)  # 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(8, 8, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

model.fit(x_train, y_train,
          epochs=1000,
          validation_split=0.2,
          callbacks=[es,],
          batch_size=1,
          )

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])
print('acc : ', result[1])
print('acc_score : ', acc_score)

'''
loss :  0.5783030986785889
acc :  0.7611111402511597
acc_score :  0.7611111111111111

>>

loss :  0.1613471806049347
acc :  0.9833333492279053
acc_score :  0.9833333333333333
'''