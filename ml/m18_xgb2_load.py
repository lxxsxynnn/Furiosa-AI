import warnings
warnings.filterwarnings('ignore')   # .dat 확장자는 불러올 때 형식을 추측했다는 경고를 냄

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor

# XGBoost 자체 방식으로 저장한 모델 불러오기 - 캘리포니아 주택 가격
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
path = './_save/xgb/'
model = XGBRegressor()
model.load_model(path + 'm18_xgb1_save.dat')

# 3. 훈련

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

'''
최종 점수:  0.8136431209957382
r2:  0.8136431209957382
'''