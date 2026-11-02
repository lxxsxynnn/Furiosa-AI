import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Kfold 실습(회귀)
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

n_split = 5
kFold = KFold(n_splits=n_split, shuffle=True, random_state=132)

# 2. 모델 구성
model = XGBRegressor()

# 3, 4. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kFold)

print("r2: ", scores, "\n평균 r2: ", round(np.mean(scores), 4))
'''
[RandomForest]
r2:  [0.81870277 0.81594018 0.80743356 0.80881228 0.79751347]
평균 r2:  0.8097

[GradientBoosting]
r2:  [0.79832998 0.79231861 0.7848689  0.79111911 0.77215835] 
평균 r2:  0.7878

[DecisionTree]
r2:  [0.60883844 0.61503394 0.61821326 0.60460988 0.6022372 ] 
평균 r2:  0.6098

[XGB]
r2:  [0.83484354 0.84412947 0.83145543 0.83406561 0.82822593] 
평균 r2:  0.8345
'''