from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import numpy as np

# CPU / GPU 실행 시간 비교 - 손글씨 숫자
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

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=100,
#     restore_best_weights=True,
# )

# save_path = 'C:/study/_save/keras34/'
# date = datetime.datetime.now()      # 현재 시간 반환
# date = date.strftime('%m%d_%H%M_')

# filename = '10_digits_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
# filepath = ''.join([save_path, 'k34_', date, filename])

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=filepath,
#                       verbose=1,
#                       )

start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          validation_split=0.2,
        #   callbacks=[es, mcp],
          batch_size=1,
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])
print('acc : ', result[1])
print('acc_score : ', acc_score)

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  1326.85 초 >> 실행 시간:  3481.89 초
'''