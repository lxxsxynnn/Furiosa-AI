from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import numpy as np

# DNN 모델을 RNN으로 바꿔보기 - 손글씨 숫자
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

x_train = x_train.reshape(-1, 64, 1)
x_test = x_test.reshape(-1, 64, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(64, 1), activation='relu'))
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
              optimizer=Adam(learning_rate=0.0015),
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=100,
    verbose=1,
    factor=0.5,
)

start = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          validation_split=0.2,
          callbacks=[es, rlr],
          batch_size=1,
          )
end = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])
print('acc : ', result[1])
print('acc_score : ', acc_score)
print("time : ", round(end - start, 2), "sec")

'''
loss :  2.033257484436035
acc :  0.5333333611488342
acc_score :  0.5333333333333333
time :  330.3 sec

>>

loss :  1.419298529624939
acc :  0.4833333194255829
acc_score :  0.48333333333333334
time :  2014.74 sec
'''