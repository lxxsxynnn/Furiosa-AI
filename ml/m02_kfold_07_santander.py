import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Kfold 실습(회귀)
# 1. 데이터
data_path = "C:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

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

[GradientBoosting]

[DecisionTree]

[XGB]
r2:  [0.20063221 0.21051478 0.20789832 0.20205206 0.20275474] 
평균 r2:  0.2048
'''