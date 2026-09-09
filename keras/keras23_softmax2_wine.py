from sklearn.datasets import load_wine
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

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

print(x_train.shape, x_test.shape)  # (142, 13) (36, 13)
print(y_train.shape, y_test.shape)  # (142, 3) (36, 3)

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

start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end=time.time()

# acc = 0.95
# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])             # loss :  0.1102718710899353
print('acc : ', round(result[1], 2))    # acc :  0.97

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)               # acc_score :  0.9722222222222222
print('걸린 시간: ', round(end - start, 2), 'sec')   # 걸린 시간:  119.74 sec