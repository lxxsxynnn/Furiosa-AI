from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np

# Drop Out 실습해보기 - 손글씨 숫자
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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
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

save_path = 'C:/study/_save/keras33/'
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

filename = '10_digits_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k33_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

model.fit(x_train, y_train,
          epochs=1000,
          validation_split=0.2,
          callbacks=[es, mcp],
          batch_size=1,
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k33_' + date + '10_digits_*.keras'))
for f in files[:-1]:
    os.remove(f)

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
loss :  0.2014123499393463
acc :  0.9722222089767456
acc_score :  0.9722222222222222

>>

loss :  0.5783030986785889
acc :  0.7611111402511597
acc_score :  0.7611111111111111
'''