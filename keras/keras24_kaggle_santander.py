# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 이진 분류 모델을 다중 분류로 바꿔보기
# 1. 데이터
path = "C:\study\_data\kaggle_santander\\"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
submit = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train.drop(['target'], axis=1)
y = train['target']

# 판다스를 넘파이로 바꾸기
# y = np.array(y)   # 방법 1
# y = y.to_numpy()  # 방법 2

y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100,
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

# 2. 모델 구성
model = Sequential()
model.add(Dense(400, input_dim=200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start = time.time()
model.fit(x_train, y_train, epochs=100,
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])             # loss :  0.24206864833831787
print('acc : ', round(result[1], 2))    # 

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)                # acc_score :  0.91105
print('time : ', round(end - start, 2), 'sec')  # time :  300.07 sec

y_submit = model.predict(test)
submit['target'] = y_submit[:, 1]

from datetime import datetime
now = datetime.now().strftime('%m%d_%H%M')
submit.to_csv(path + "submit/" + "submit_" + now + ".csv")