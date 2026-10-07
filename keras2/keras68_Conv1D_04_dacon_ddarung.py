from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

# DNN 모델에 Conv1D 적용해보기 - 따릉이(데이콘)
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
test_csv = pd.read_csv(data_path + "test.csv", index_col= 0)

#여기에 model.predict(submission) 값을 넣음
submission = pd.read_csv(data_path + "submission.csv")

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=333)

x_train = x_train.values.reshape(-1, 9, 1)
x_test = x_test.values.reshape(-1, 9, 1)
# 2. 모델 구성
model = Sequential()
model.add(Conv1D(32, kernel_size=3, input_shape=(9, 1)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    factor=0.4,
)

model.compile(loss="mse", optimizer=Adam(learning_rate=0.005))
start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[es, rlr]
          )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)

print("loss : ", loss)                          # loss :  1937.4571533203125 > 2945.165771484375
print("r2 : ", r2)                              # r2 :  0.7123152845140399 > 0.5626848739978303
print("mse : ", mse)                            # mse :  1937.456993767232 > 2945.1660228867877
print("RMSE : ", rmse)                          # RMSE :  44.01655363345967 > 54.269383844731344
print("time : ", round(end - start, 2), "sec")  # time :  13.27 sec > 14.96 sec