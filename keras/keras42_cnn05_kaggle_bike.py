# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 2차원 데이터를 CNN 모델로 처리해보기 - 자전거(캐글)
# 1. 데이터
data_path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
submission = pd.read_csv(data_path + "sampleSubmission.csv", index_col=0)
# print(train_csv.describe())

############## 결측치 확인 ##############
print(train_csv.isna().sum())   # 결측치가 있는지 확인 info랑 둘 중에 편한 거 골라서 사용하면 됨
print(test_csv.isnull().sum())  # 마찬가지로 결측치 확인하는 방법

############## x, y 분리 ##############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # test에 없는 컬럼 제거
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']  # train_csv에서 count만 사용
print(y, y.shape)       # (10886,) > 벡터 반환

# test, train 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=23)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 4, 2, 1)
x_test = x_test.reshape(-1, 4, 2, 1)

print(x_train.shape, x_test.shape)  # 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(4, 2, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=100,
                   restore_best_weights=True,
                   verbose=1
                   )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2,
                callbacks=[es,],
                 )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("loss : ", loss)
print("r2 : ", r2)
print("mse : ", mse)
print("rmse : ", rmse)

'''
loss :  26070.4765625
r2 :  0.2429511547088623
mse :  26070.478515625
rmse :  161.4635516629837

>>

loss :  22827.75390625
r2 :  0.33711516857147217
mse :  22827.75390625
rmse :  151.0885631219319
'''

# y_submit = model.predict(test_csv)
# submission['count'] = y_submit

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission.to_csv(data_path + "submit/" + "submit_" + now + ".csv")