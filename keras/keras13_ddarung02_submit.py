# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
# path = "./_data/ddarung/"            # 상대경로
# path = "C:\study\_data/ddarung/"     # 절대경로
# path = "C:\study\_data\ddarung/"     # '/', '\' 상관없음
# path = "C:\\study//_data//ddarung/"
# path = "C:\\study\\_data\\ddarung\\" # '//', '\\'도 가능
path = "C:\\study\_data\\ddarung/"    # 섞어써도 되는데 가급적 하나만 사용 권장

 
train_csv = pd.read_csv(path + "train.csv", index_col=0) # index_col - 인덱스 컬럼(제외)
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "submission.csv")

print(train_csv)
print(train_csv.shape)      # [1459 rows x 10 columns]
print(train_csv.columns)    # (1459, 10)
print(train_csv.info())     # pandas에 있는 info값 출력

############ 결측치 처리 1. 삭제 ############
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

########### train_csv를 x와 y로 분리 ###########
x = train_csv.drop(['count'], axis =1) # 열(컬럼) 삭제
print("x : ", x) # x : [1328 rows x 9 columns] 

y = train_csv['count'] 
print(y)
print(y.shape) # (1328,) -> y가 벡터 형태로 빠짐


test_csv = pd.read_csv(path + "test.csv", index_col= 0) # 행 > 성능을 좌우하는 데이터의 양 / 모델링할 때는 열이 중요, 열 = 컬럼 = 특성 = 속성 = 피쳐 = 어트리뷰트
print(test_csv)         # [715 rows x 9 columns] 
# print(test_csv.shape)   # (715, 9)
# print(test_csv.columns)
# print(test_csv.info())


submission = pd.read_csv(path + "submission.csv" , index_col= 0) #여기에 model.predict(submission) 값을 넣음
# print(submission) # [715 * 9] / NaN: 결측치 
# print(submission.shape)
# print(submission.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, random_state=1)

# print('x_train : ', x_train.shape)
# print('x_test : ', x_test.shape)
# print('y_train : ', y_train.shape)
# print('y_test : ', y_test.shape)

############# submit 물밑작업 #############
# print(test_csv.info())

# 제출용 데이터 > 결측치 처리를 삭제로 하면 안됨 > 평균값으로 채울거임
############ 결측치 처리 2. 평균값 넣기 ############
test_csv = test_csv.fillna(test_csv.mean()) # pandas dataset 형태 / fillna() : fill + na 결측지를 채움 / mean{} : 평균치 함수
# print(test_csv.info())
# print(test_csv.shape)   # (715, 9)




# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.2)

#4. 평가, 예측
print("========================================")
y_predict = model.predict(x_test, batch_size=16)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

############# submission.csv 만들기 // count 컬럼에 값 넣어줌 #############
# print(submission)
y_submit = model.predict(test_csv)
submission['count'] = y_submit  # count열에 y_submit 값을 채워넣음(pandas 문법)
# print(submission)           # 훈련시켜서 최종 w를 구했고, 그걸 기반으로 test_scv에 대한 예측값을 구해서 submit의 count에 채워넣음
# print(submission.shape)     # (715, 1)

submission.to_csv(path + "/submit/" + "submit_0904_1443.csv")

"""
r2:  0.4348125892037864
mse:  3680.4407978878844
RMSE :  60.66663661262164

r2:  0.47410571442065963
mse:  3979.077398012542
RMSE :  63.079928646222655

r2:  0.5941688149084157
mse:  3012.3796722921247
RMSE :  54.88514983392252

r2:  0.6433370017594988
mse:  2647.4169685012885
RMSE :  51.45305596853591

r2:  0.6327886825281774
mse:  2364.7114087433624
RMSE :  48.628298435616294

r2:  0.5426588515491175
mse:  3460.3833448718033
RMSE :  58.82502311832783
"""