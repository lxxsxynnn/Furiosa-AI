import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GlobalAveragePooling1D, Input, Concatenate 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 앙상블 모델로 삼성전자 주가 맞추기 - 모델 저장
# 1. 데이터
# 컬럼 5개 이상 사용
path = 'C:/study/_data/samsung/'

def load_csv(name):
    df = pd.read_csv(path + name, index_col=0, encoding='cp949', thousands=',')  # 쉼표 숫자를 숫자로 읽음
    df.index = pd.to_datetime(df.index)     # '2026/10/12' 문자열 → 날짜
    return df.sort_index()                  # 과거 → 최신순

x1_data = load_csv('아모레261012.csv')
x2_data = load_csv('하이닉스261012.csv')
y_data = load_csv('삼성전자261012.csv')

print(x1_data.shape) # (2060, 16)
print(x2_data.shape) # (1300, 16)
print(y_data.shape)  # (1120, 16)

# 사용할 컬럼
cols = ['시가', '고가', '저가', '종가', '거래량', '기관', '외인(수량)']

# 세 파일에 모두 있는 날짜만 (가장 짧은 삼성전자 기간에 맞추기)
dates = x1_data.index.intersection(x2_data.index).intersection(y_data.index)
x1_data = x1_data.loc[dates, cols]
x2_data = x2_data.loc[dates, cols]

# y : 다음 날 삼성전자 시가(하루씩 당겨서 오늘 행에 내일 시가를 붙임 / 마지막 날은 NaN)
y_data = y_data.loc[dates, '시가'].shift(-1)

x1 = x1_data.values     # (1120, 7)
x2 = x2_data.values     # (1120, 7)
y = y_data.values       # (1120,)

# timesteps일씩 잘라 다음 날 시가를 맞춤
timesteps = 5

def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size)]
        arr.append(subset)
    return np.array(arr)

# 스케일링 : 섞어서 나누면 train 날짜가 전체 기간에 흩어지므로 전체로 fit
scaler1 = MinMaxScaler()
scaler1.fit(x1)
x1 = scaler1.transform(x1)

scaler2 = MinMaxScaler()                # 종목마다 값 범위가 달라서 별개의 스케일러 사용
scaler2.fit(x2)
x2 = scaler2.transform(x2)

x1 = split_x(x1, timesteps)             # (1115, 5, 7)
x2 = split_x(x2, timesteps)             # (1115, 5, 7)
y = y[timesteps - 1:]                   # 각 묶음의 마지막 날 다음 날 시가 (1115,)

# 마지막 묶음(~2026/10/12)은 내일 시가가 없어서 학습에서 빼고, 내일 예측할 입력으로 씀
x1_pred, x2_pred = x1[-1:], x2[-1:]     # (1, 5, 7)
x1, x2, y = x1[:-1], x2[:-1], y[:-1]

print(x1.shape, x2.shape, y.shape)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y,
    shuffle=True,       # 순서대로 자르면 최근 고가 구간(train 최고가의 3배)이 train에 없어 그 값을 못 냄
    random_state=333,
    train_size=0.8,
)

print(x1_train.shape)   # (892, 5, 7)
print(x1_test.shape)    # (223, 5, 7)
print(x2_train.shape)   # (892, 5, 7)
print(x2_test.shape)    # (223, 5, 7)
print(y_train.shape)    # (892,)
print(y_test.shape)     # (223,)

# 2. 모델 구성
# LSTM, Conv1D 사용
# 2-1. 모델
input1 = Input(shape=(5, 7))
lstm11 = LSTM(32, return_sequences=True)(input1)
lstm12 = LSTM(64, return_sequences=True)(lstm11)
conv1 = Conv1D(32, 3)(lstm12)
gap1 = GlobalAveragePooling1D()(conv1)
output1 = Dense(100, activation='relu')(gap1)

# 2-2. 모델
input2 = Input(shape=(5, 7))
lstm21 = LSTM(32, return_sequences=True)(input2)
lstm22 = LSTM(64, return_sequences=True)(lstm21)
conv2 = Conv1D(32, 3)(lstm22)
gap2 = GlobalAveragePooling1D()(conv2)
output2 = Dense(128, activation='relu')(gap2)

# 2-3. 모델 합치기
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(5, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)

last_output = Dense(1, name='last')(merge3)
model = Model(inputs=(input1, input2), outputs=last_output)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

save_path='C:/study/_save/keras70/'
os.makedirs(save_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
    filepath=save_path + 'keras70.weights.h5',
)

model.fit([x1_train, x2_train], y_train,
          epochs=5000,
          validation_split=0.1,
          callbacks=[es, mcp],
          batch_size=16,
          )

model.save(save_path + 'keras70_model.keras')

# 내일 삼성전자 시가 맞추기
# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print("loss: ", loss)   # 112517352.0

y_predict = model.predict([x1_pred, x2_pred])
print('내일 삼성전자 시가: ', y_predict[0][0])  # 내일 삼성전자 시가:  249431.69