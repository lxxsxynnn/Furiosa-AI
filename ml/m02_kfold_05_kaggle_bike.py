import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Kfold 실습(회귀)
# 1. 데이터
path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(path + "test.csv", index_col=0)
train_csv = train_csv.drop(['casual', 'registered'], axis=1)   # test에 없는 컬럼 제거

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']  # train_csv에서 count만 사용

n_split = 5
kFold = KFold(n_splits=n_split, shuffle=True, random_state=132)

# 2. 모델 구성
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
model = XGBRegressor()

# 3, 4. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kFold)

print("r2: ", scores, "\n평균 r2: ", round(np.mean(scores), 4))
'''
[RandomForest]
r2:  [0.30151113 0.28628721 0.30559335 0.30676009 0.27934024] 
평균 r2:  0.2959

[GradientBoosting]
r2:  [0.33355553 0.31843573 0.35098829 0.34147982 0.31666338] 
평균 r2:  0.3322

[DecisionTree]
r2:  [-0.09854208 -0.08052475 -0.18859608 -0.19550145 -0.17134867] 
평균 r2:  -0.1469

[XGB]
r2:  [0.33798259 0.30633509 0.35815352 0.33848614 0.30906701] 
평균 r2:  0.337
'''