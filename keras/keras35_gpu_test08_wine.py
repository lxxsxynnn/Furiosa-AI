from sklearn.datasets import load_wine
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# CPU / GPU 실행 시간 비교 - 와인
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
model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc']
              )
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=100,
#     restore_best_weights=True,
# )

# save_path = 'C:/study/_save/keras34/'
# date = datetime.datetime.now()      # 현재 시간 반환
# date = date.strftime('%m%d_%H%M_')

# filename = '08_wine_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
# filepath = ''.join([save_path, 'k34_', date, filename])

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=filepath,
#                       verbose=1,
#                       )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
        #   callbacks=[es, mcp],
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', result[0])
print('acc : ', round(result[1], 2))
print('acc_score : ', acc_score)

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  243.65 초 >> 실행 시간:  281.01 초
'''