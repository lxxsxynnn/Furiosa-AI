from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

save_path = 'C:/study/_save/keras31/'

# 최고 가중치 불러오기 - 따릉이(데이콘)
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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
model = load_model(save_path + 'k31_0914_1509_04_ddarung_0052-3175.8865.keras')

# 3. 컴파일, 훈련

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  2854.98193359375

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)          # r2 :  0.5760759078633605

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)        # mse :  2854.9820443159433

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)      # RMSE :  53.43203200624082

'''
loss :  2854.98193359375
r2 :  0.5760759078633605
mse :  2854.9820443159433
RMSE :  53.43203200624082
'''