from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
import numpy as np

# 2차원 데이터를 CNN 모델로 처리해보기 - 따릉이(데이콘)
# 1. 데이터
data_path = "C:/study/_data/ddarung/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0) # index_col - 인덱스 컬럼(제외)

############ 결측치 처리 1.삭제 ############
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

########### train_csv를 x와 y로 분리 ###########
x = train_csv.drop(['count'], axis=1) # count 컬럼을 제거한 나머지 / axis=1 열(컬럼) 삭제
print("x : ", x) # x : [1328 rows x 9 columns] 

y = train_csv['count'] # count 컬럼만 선택
print(y)
print(y.shape) # (1328,) -> y가 벡터 형태로 빠짐

# model.predict()에 넣을 값
test_csv = pd.read_csv(data_path + "test.csv", index_col= 0) # id 컬럼은 순번(인덱스)인데 이걸 불러오면 id가 실제 데이터처럼 포함됨 -> index_col=0으로 인덱스 지정하면 데이터에서 제외됨

#여기에 model.predict(submission) 값을 넣음
submission = pd.read_csv(data_path + "submission.csv")

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=333)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 3, 3, 1)
x_test = x_test.reshape(-1, 3, 3, 1)

print(x_train.shape, x_test.shape)  # (1062, 3, 3, 1) (266, 3, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(3, 3, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
                   )

hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[es, ]
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("mse : ", mse)
print("RMSE : ", rmse)

'''
loss :  2884.355224609375
r2 :  0.5717144546277286
mse :  2884.3549223991245
RMSE :  53.706190726946225

>>

loss :  1631.1834716796875
r2 :  0.7577925297136118
mse :  1631.1834866038746
RMSE :  40.38791262994257
'''