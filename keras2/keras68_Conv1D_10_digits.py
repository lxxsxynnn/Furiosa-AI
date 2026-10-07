from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import numpy as np

# DNN 모델에 Conv1D 적용해보기 - 손글씨 숫자
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
    random_state=21,
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
# model.add(LSTM(100, input_shape=(64, 1), activation='relu'))
model.add(Conv1D(72, kernel_size=6, input_shape=(64, 1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
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

print('loss : ', result[0])                     # loss :  1.419298529624939 > 0.17294864356517792
print('acc : ', result[1])                      # acc :  0.4833333194255829 > 0.9583333134651184
print('acc_score : ', acc_score)                # acc_score :  0.48333333333333334 > 0.9583333333333334
print("time : ", round(end - start, 2), "sec")  # time :  2014.74 sec > 493.07 sec

'''
cf) GAP 적용 시
loss :  2.302415370941162
acc :  0.12222222238779068
acc_score :  0.12222222222222222
time :  231.27 sec
'''