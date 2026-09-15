from sklearn.datasets import load_wine
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# 최고 가중치 저장하기 - 와인
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

# 2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

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

save_path = 'C:/study/_save/keras31/'
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

filename = '08_wine_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k31_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k31_' + date + '08_wine_*.keras'))
for f in files[:-1]:
    os.remove(f)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1], 2))

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', acc_score)

'''
loss :  0.1525217890739441
acc :  0.97
acc_score :  0.9722222222222222
'''