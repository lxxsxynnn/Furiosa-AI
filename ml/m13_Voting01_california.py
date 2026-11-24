import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# 종류가 다른 모델 네 개를 보팅으로 묶어 보기 - 캘리포니아 주택 가격
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
xgb = XGBRegressor()
lgbm = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('XGB', xgb), ('LGBM', lgbm), ('RF', rf), ('CAT', cat)],
    # voting='hard',  # Default
    # voting='soft',
    # Regressor는 voting parameter가 없음
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

'''
최종 점수:  0.8406838952754865
r2:  0.8406838952754865
'''