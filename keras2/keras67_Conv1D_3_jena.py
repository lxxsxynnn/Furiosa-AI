# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data
import numpy as np
import pandas as pd
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# RNN 모델을 CNN으로 바꿔보기(Conv1D 적용)
# 12월 31일 00시 10분부터 1월 1일 00시 00분까지의 T(degC) 맞추기

# 1. 데이터
path = "C:/study/_data/kaggle_jena/"

data = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
print(data.shape)    # (420551, 14)
y_answer = data[- 144:][data.columns[1]]
print(y_answer.shape)    # (144,)

x_data = data[:-288].drop(data.columns[1], axis=1)
y_data = data[144:-144][data.columns[1]]

print(x_data.shape)  # (420263, 13)
print(y_data.shape)  # (420263,)

size = 144

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)   # 창으로 나누기 전, 2차원일 때 스케일링

def split(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        arr.append(subset)
    return np.array(arr)

split_start = time.time()
x = split(x_data, size)
y = y_data.to_numpy()[size - 1:]    # split(y_data, size)[:, -1]과 같은 값
split_end = time.time()

print('데이터 가공 소요 시간: ', round(split_end - split_start, 2), '초')   # 데이터 가공 소요 시간:  1.38 초
print(x.shape)  # (420120, 144, 13)
print(y.shape)  # (420120,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=23,
                                                    )
print(x_train.shape)    # (336096, 144, 13)
print(x_test.shape)     # (84024, 144, 13)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(10, kernel_size=3, input_shape=(144, 13)))
model.add(Flatten())
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=100,
)

file_path='C:/study/_save/'
os.makedirs(file_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=file_path + 'keras67_jenna.keras',
)

fit_start = time.time()
model.fit(x_train, y_train, epochs=5000,
          callbacks=[es, mcp],
          batch_size=200,
          validation_split=0.2,
          )
fit_end = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results)   # loss :  7.214554786682129 > 10.205568313598633

x_predict = data[-431:-144].drop(data.columns[1], axis=1)
x_predict = scaler.transform(x_predict)     # 훈련과 같은 스케일러
x_predict = split(x_predict, size)
y_predict = model.predict(x_predict)

r2 = r2_score(y_answer, y_predict)
print('r2 : ', r2)      # r2 :  0.5035395063265995 > 0.575290404720308

mse = mean_squared_error(y_answer, y_predict)
print('mse : ', mse)    # mse :  5.408995168950548 > 4.6272607354454625

def RMSE(y_answer, y_predict):
    return np.sqrt(mean_squared_error(y_answer, y_predict))

rmse = RMSE(y_answer, y_predict)
print("RMSE : ", rmse)  # RMSE :  2.3257246545862964 > 2.1511068628604817

print('훈련 소요 시간 : ', round(fit_end - fit_start, 2), '초')  # 훈련 소요 시간 :  3613.48 초 > 1896.7 초